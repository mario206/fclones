use std::fs;
use std::fs::Metadata;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::collections::VecDeque;

use filetime::FileTime;

use crate::dedupe::{FsCommand, PathAndMetadata};
use crate::log::{Log, LogExt};
use crate::config::ReflinkMode;

/// A shared queue for tree-shaped deduplication.
///
/// In safe mode (FIDEDUPERANGE), the kernel acquires i_rwsem on the src file,
/// which serializes all dedup operations sharing the same src. This queue allows
/// completed dest files to become new src candidates, enabling concurrent dedup
/// with exponentially growing parallelism.
///
/// Flow:
/// 1. Queue starts with only the retained file.
/// 2. Each worker takes a src from the queue (blocks if empty).
/// 3. After successful FIDEDUPERANGE, both src and dest are put back into the queue.
/// 4. After failure, only src is put back.
/// 5. Workers signal completion; when all are done, remaining waiters are unblocked.
pub struct DedupeQueue {
    inner: Mutex<DedupeQueueInner>,
    condvar: Condvar,
}

impl std::fmt::Debug for DedupeQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("DedupeQueue")
            .field("queue_len", &inner.queue.len())
            .field("total_workers", &inner.total_workers)
            .field("completed_workers", &inner.completed_workers)
            .finish()
    }
}

struct DedupeQueueInner {
    queue: VecDeque<PathBuf>,
    /// Total number of workers that will use this queue
    total_workers: usize,
    /// Number of workers that have completed
    completed_workers: usize,
}

impl DedupeQueue {
    /// Create a new queue with the retained file as the initial src.
    pub fn new(retained: PathBuf, total_workers: usize) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(retained);
        Self {
            inner: Mutex::new(DedupeQueueInner {
                queue,
                total_workers,
                completed_workers: 0,
            }),
            condvar: Condvar::new(),
        }
    }

    /// Take an available src from the queue.
    /// Blocks if the queue is empty and there are still active workers.
    /// Returns None if all workers have completed and the queue is empty.
    /// On success, returns (src_path, remaining_queue_length).
    fn take(&self) -> Option<(PathBuf, usize)> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(src) = inner.queue.pop_front() {
                let remaining = inner.queue.len();
                return Some((src, remaining));
            }
            // If all workers are done, no more src will be added
            if inner.completed_workers >= inner.total_workers {
                return None;
            }
            inner = self.condvar.wait(inner).unwrap();
        }
    }

    /// Put src back and optionally add a successfully deduped dest.
    fn put_back(&self, src: PathBuf, dest: Option<PathBuf>) {
        let mut inner = self.inner.lock().unwrap();
        inner.queue.push_back(src);
        if let Some(d) = dest {
            inner.queue.push_back(d);
        }
        // Notify all waiting workers that new src(s) are available
        self.condvar.notify_all();
    }

    /// Mark a worker as completed. When all workers are done, wake up any
    /// remaining waiters so they can exit.
    fn mark_completed(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.completed_workers += 1;
        if inner.completed_workers >= inner.total_workers {
            self.condvar.notify_all();
        }
    }
}

/// Format byte size into human-readable string (B / MB / GB)
fn format_size(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(unix)]
struct XAttr {
    name: std::ffi::OsString,
    value: Option<Vec<u8>>,
}

/// Calls OS-specific reflink implementations with an option to call the more generic
/// one during testing one on Linux ("crosstesting").
/// The destination file is allowed to exist.
pub fn reflink(
    src: &PathAndMetadata,
    dest: &PathAndMetadata,
    mode: Option<ReflinkMode>,
    queue: Option<&Arc<DedupeQueue>>,
    log: &dyn Log,
) -> io::Result<()> {
    // Remember original metadata of the parent directory:
    let dest_parent = dest.path.parent();
    let dest_parent_metadata = dest_parent.map(|p| p.to_path_buf().metadata());

    // Call reflink:
    let result = || -> io::Result<()> {
        let dest_path_buf = dest.path.to_path_buf();

        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if !crosstest() {
                // On Linux, mode must be specified (fast or safe)
                let mode = mode.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "On Linux, --reflink-mode must be specified (fast or safe)",
                    )
                })?;
                match mode {
                    ReflinkMode::Fast => {
                        linux_reflink_fast(src, dest, log)?;
                        // FICLONE may modify mtime, so restore it
                        return restore_metadata(&dest_path_buf, &dest.metadata, Restore::TimestampOnly);
                    }
                    ReflinkMode::Safe => {
                        linux_reflink_safe(src, dest, queue, log)?;
                        // FIDEDUPERANGE does not modify mtime, so no need to restore it
                        return Ok(());
                    }
                }
            }
        }

        // Non-Linux platforms or crosstest mode
        #[cfg(unix)]
        let dest_xattrs = get_xattrs(&dest_path_buf)?;

        // Suppress unused variable warning on non-Linux platforms
        let _ = mode;
        let _ = queue;

        safe_reflink(src, dest, log)?;

        #[cfg(unix)]
        restore_xattrs(&dest_path_buf, dest_xattrs)?;

        restore_metadata(
            &dest_path_buf,
            &dest.metadata,
            Restore::TimestampOwnersPermissions,
        )
    }()
    .map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to deduplicate {dest} -> {src}: {e}"),
        )
    });

    // Restore the original metadata of the deduplicated files's parent directory:
    if let Some(parent) = dest_parent {
        if let Some(metadata) = dest_parent_metadata {
            let result = metadata.and_then(|metadata| {
                restore_metadata(&parent.to_path_buf(), &metadata, Restore::TimestampOnly)
            });
            if let Err(e) = result {
                log.warn(format!(
                    "Failed keep metadata for {}: {}",
                    parent.display(),
                    e
                ))
            }
        }
    }

    result
}

// First reflink (not move) the target file out of the way (this also checks for
// reflink support), then overwrite the existing file to preserve most metadata and xattrs.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_reflink_fast(src: &PathAndMetadata, dest: &PathAndMetadata, log: &dyn Log) -> io::Result<()> {
    let tmp = FsCommand::temp_file(&dest.path);
    let std_tmp = tmp.to_path_buf();

    let fs_target = src.path.to_path_buf();
    let std_link = dest.path.to_path_buf();

    let remove_temporary = |temporary| {
        if let Err(e) = FsCommand::remove(&temporary) {
            log.warn(format!(
                "Failed to remove temporary {}: {}",
                temporary.display(),
                e
            ))
        }
    };

    // Backup via reflink, if this fails then the fs does not support reflinking.
    if let Err(e) = reflink_overwrite(&std_link, &std_tmp) {
        remove_temporary(tmp);
        return Err(e);
    }

    // Use FICLONE (does not verify content identity)
    let result = reflink_overwrite(&fs_target, &std_link);

    match result {
        Err(e) => {
            if let Err(remove_err) = FsCommand::unsafe_rename(&tmp, &dest.path) {
                log.warn(format!(
                    "Failed to undo deduplication from {} to {}: {}",
                    &dest,
                    tmp.display(),
                    remove_err
                ))
            }
            Err(e)
        }
        Ok(ok) => {
            remove_temporary(tmp);
            Ok(ok)
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_reflink_safe(
    src: &PathAndMetadata,
    dest: &PathAndMetadata,
    queue: Option<&Arc<DedupeQueue>>,
    log: &dyn Log,
) -> io::Result<()> {
    let std_link = dest.path.to_path_buf();

    if let Some(q) = queue {
        // Tree-shaped deduplication: take an available src from the shared queue.
        // This src may be the original retained file or any previously deduped file.
        let actual_src = match q.take() {
            Some((s, remaining)) => {
                log.verbose(format!(
                    "DedupeQueue: took src={}, dest={}, remaining_in_queue={}",
                    s.display(),
                    std_link.display(),
                    remaining
                ));
                s
            }
            None => {
                // All workers completed and queue is empty - should not happen
                // in normal flow, but handle gracefully.
                q.mark_completed();
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "DedupeQueue exhausted unexpectedly",
                ));
            }
        };

        let result = reflink_overwrite_dedupe(&actual_src, &std_link, log);

        match &result {
            Ok(()) => {
                // Success: put src back and add dest as a new potential src
                q.put_back(actual_src, Some(std_link));
            }
            Err(_) => {
                // Failure: put src back but don't add dest
                q.put_back(actual_src, None);
            }
        }
        q.mark_completed();
        result
    } else {
        // No queue: use original src directly (backward compatible)
        let fs_target = src.path.to_path_buf();
        reflink_overwrite_dedupe(&fs_target, &std_link, log)
    }
}

/// Reflink `target` to `link` and expect these two files to be equally sized.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn reflink_overwrite(target: &std::path::Path, link: &std::path::Path) -> io::Result<()> {
    use nix::request_code_write;
    use std::os::unix::prelude::AsRawFd;

    let src = fs::File::open(target)?;

    // This operation does not require `.truncate(true)` because the files are already of the same size.
    let dest = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(link)?;

    // From /usr/include/linux/fs.h:
    // #define FICLONE		_IOW(0x94, 9, int)
    const FICLONE_TYPE: u8 = 0x94;
    const FICLONE_NR: u8 = 9;
    const FICLONE_SIZE: usize = std::mem::size_of::<libc::c_int>();

    let ret = unsafe {
        libc::ioctl(
            dest.as_raw_fd(),
            request_code_write!(FICLONE_TYPE, FICLONE_NR, FICLONE_SIZE),
            src.as_raw_fd(),
        )
    };

    #[allow(clippy::if_same_then_else)]
    if ret == -1 {
        let err = io::Error::last_os_error();
        let code = err.raw_os_error().unwrap(); // unwrap () Ok, created from `last_os_error()`
        if code == libc::EOPNOTSUPP { // 95
             // Filesystem does not supported reflinks.
             // No cleanup required, file is left untouched.
        } else if code == libc::EINVAL { // 22
             // Source filesize was larger than destination.
        }
        Err(err)
    } else {
        Ok(())
    }
}

/// New implementation using FIDEDUPERANGE for safer deduplication
#[cfg(any(target_os = "linux", target_os = "android"))]
fn reflink_overwrite_dedupe(target: &std::path::Path, link: &std::path::Path, log: &dyn Log) -> io::Result<()> {
    use nix::request_code_readwrite;
    use std::mem::{size_of, zeroed};
    use std::os::unix::prelude::AsRawFd;

    let src = fs::File::open(target)?;
    let src_metadata = src.metadata()?;
    let src_size = src_metadata.len();

    let start_time = std::time::Instant::now();

    log.verbose(format!(
        "Safe dedup: {} -> {} ({})",
        target.display(),
        link.display(),
        format_size(src_size)
    ));

    // This operation does not require `.truncate(true)` because the files are already of the same size.
    let dest = fs::OpenOptions::new()
        .create(false)
        .truncate(false)
        .write(true)
        .open(link)?;

    // FIDEDUPERANGE only deduplicates the range specified by src_length,
    // so if files have different sizes, some bytes would be left un-deduped.
    // Reject early to avoid silent partial deduplication.
    let dest_metadata = dest.metadata()?;
    let dest_size = dest_metadata.len();
    if src_size != dest_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "File sizes differ (source={} bytes, dest={} bytes), cannot deduplicate with FIDEDUPERANGE",
                src_size, dest_size
            ),
        ));
    }

    // From /usr/include/linux/fs.h:
    // #define FIDEDUPERANGE _IOWR(0x94, 54, struct file_dedupe_range)
    const FIDEDUPERANGE_TYPE: u8 = 0x94;
    const FIDEDUPERANGE_NR: u8 = 54;

    // Status codes from Linux kernel
    // FILE_DEDUPE_RANGE_SAME = 0: Blocks are identical and were successfully deduplicated
    const FILE_DEDUPE_RANGE_DIFFERS: i32 = 1;

    // Define dedupe range structures
    #[repr(C)]
    struct FileDedupRangeInfo {
        dest_fd: i64,
        dest_offset: u64,
        bytes_deduped: u64,
        status: i32,
        reserved: u32,
    }

    // Header-only struct matching the kernel's struct file_dedupe_range
    // (with flexible array member info[0], which has zero size).
    // Used only for computing the correct ioctl number.
    #[repr(C)]
    struct FileDedupRangeHeader {
        src_offset: u64,
        src_length: u64,
        dest_count: u16,
        reserved1: u16,
        reserved2: u32,
    }

    #[repr(C)]
    struct FileDedupRange {
        src_offset: u64,
        src_length: u64,
        dest_count: u16,
        reserved1: u16,
        reserved2: u32,
        info: [FileDedupRangeInfo; 1],
    }

    // The ioctl number must be encoded with the size of the header only
    // (without the flexible array member), matching the kernel definition:
    //   #define FIDEDUPERANGE _IOWR(0x94, 54, struct file_dedupe_range)
    // where sizeof(struct file_dedupe_range) == 24 (info[0] has zero size).
    const FIDEDUPERANGE_SIZE: usize = size_of::<FileDedupRangeHeader>();

    // Process deduplication potentially in chunks
    // Prior to Linux kernel 4.18, btrfs had a 16MiB restriction on FIDEDUPERANGE
    // This loop handles both older kernels (multiple iterations) and newer ones (likely one iteration)
    let mut offset: u64 = 0;

    while offset < src_size {
        // Prepare dedupe range struct
        let mut dedupe_range: FileDedupRange = unsafe { zeroed() };

        // Set source information
        dedupe_range.src_offset = offset;
        dedupe_range.src_length = src_size - offset;
        dedupe_range.dest_count = 1;
        dedupe_range.reserved1 = 0;
        dedupe_range.reserved2 = 0;

        // Set destination information
        dedupe_range.info[0].dest_fd = i64::from(dest.as_raw_fd());
        dedupe_range.info[0].dest_offset = offset;
        dedupe_range.info[0].bytes_deduped = 0;
        dedupe_range.info[0].status = 0;
        dedupe_range.info[0].reserved = 0;

        // Call FIDEDUPERANGE ioctl
        let ret = unsafe {
            libc::ioctl(
                src.as_raw_fd(),
                request_code_readwrite!(FIDEDUPERANGE_TYPE, FIDEDUPERANGE_NR, FIDEDUPERANGE_SIZE)
                    as libc::c_ulong,
                &mut dedupe_range,
            )
        };

        #[allow(clippy::if_same_then_else)]
        if ret == -1 {
            let err = io::Error::last_os_error();
            let code = err.raw_os_error().unwrap(); // unwrap () Ok, created from `last_os_error()`
            if code == libc::EOPNOTSUPP { // 95
                 // Filesystem does not supported reflinks.
                 // No cleanup required, file is left untouched.
            } else if code == libc::EINVAL { // 22
                 // Source filesize was larger than destination.
            }
            return Err(err);
        }

        // Check for content differences - FIDEDUPERANGE verifies content identity
        if dedupe_range.info[0].status == FILE_DEDUPE_RANGE_DIFFERS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File contents differ, cannot deduplicate",
            ));
        }

        // Get bytes deduped - on older btrfs (pre-kernel 4.18), this may be limited to 16MiB
        // On newer kernels, this will typically process the entire file in one go
        let bytes_deduped = dedupe_range.info[0].bytes_deduped;
        if bytes_deduped == 0 {
            // No bytes deduped but no error, might be end of file
            break;
        }

        // Move offset for next chunk
        offset += bytes_deduped;
    }

    let elapsed = start_time.elapsed();
    log.verbose(format!(
        "Safe dedup done: {} -> {} ({}) in {:.3}s",
        target.display(),
        link.display(),
        format_size(src_size),
        elapsed.as_secs_f64()
    ));

    Ok(())
}

/// Restores file owner and group
#[cfg(unix)]
fn restore_owner(path: &std::path::Path, metadata: &Metadata) -> io::Result<()> {
    use file_owner::PathExt;
    use std::os::unix::fs::MetadataExt;

    let uid = metadata.uid();
    let gid = metadata.gid();
    path.set_group(gid).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to set file group of {}: {}", path.display(), e),
        )
    })?;
    path.set_owner(uid).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to set file owner of {}: {}", path.display(), e),
        )
    })?;
    Ok(())
}

#[derive(Debug, PartialEq)]
enum Restore {
    TimestampOnly,
    TimestampOwnersPermissions,
}

// Not kept: xattrs, ACLs, etc.
fn restore_metadata(
    path: &std::path::Path,
    metadata: &Metadata,
    restore: Restore,
) -> io::Result<()> {
    let atime = FileTime::from_last_access_time(metadata);
    let mtime = FileTime::from_last_modification_time(metadata);

    filetime::set_file_times(path, atime, mtime).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "Failed to set access and modification times for {}: {}",
                path.display(),
                e
            ),
        )
    })?;

    if restore == Restore::TimestampOwnersPermissions {
        fs::set_permissions(path, metadata.permissions()).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to set permissions for {}: {}", path.display(), e),
            )
        })?;

        #[cfg(unix)]
        restore_owner(path, metadata)?;
    }
    Ok(())
}

#[cfg(unix)]
fn get_xattrs(path: &std::path::Path) -> io::Result<Vec<XAttr>> {
    use itertools::Itertools;
    use xattr::FileExt;

    let file = fs::File::open(path)?;
    file.list_xattr()
        .map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "Failed to list extended attributes of {}: {}",
                    path.display(),
                    e
                ),
            )
        })?
        .map(|name| {
            Ok(XAttr {
                value: file.get_xattr(name.as_os_str()).map_err(|e| {
                    io::Error::new(
                        e.kind(),
                        format!(
                            "Failed to read extended attribute {} of {}: {}",
                            name.to_string_lossy(),
                            path.display(),
                            e
                        ),
                    )
                })?,
                name,
            })
        })
        .try_collect()
}

#[cfg(unix)]
fn restore_xattrs(path: &std::path::Path, xattrs: Vec<XAttr>) -> io::Result<()> {
    use xattr::FileExt;
    let file = fs::File::open(path)?;
    for name in file.list_xattr()? {
        file.remove_xattr(&name).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "Failed to clear extended attribute {} of {}: {}",
                    name.to_string_lossy(),
                    path.display(),
                    e
                ),
            )
        })?;
    }
    for attr in xattrs {
        if let Some(value) = attr.value {
            file.set_xattr(&attr.name, &value).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to set extended attribute {} of {}: {}",
                        attr.name.to_string_lossy(),
                        path.display(),
                        e
                    ),
                )
            })?;
        }
    }
    Ok(())
}

// Reflink which expects the destination to not exist.
#[cfg(any(not(any(target_os = "linux", target_os = "android")), test))]
fn copy_by_reflink(src: &crate::path::Path, dest: &crate::path::Path) -> io::Result<()> {
    reflink::reflink(src.to_path_buf(), dest.to_path_buf())
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to reflink: {e}")))
}

// Create a reflink by removing the file and making a reflink copy of the original.
// After successful copy, attempts to restore the metadata of the file.
// If reflink or metadata restoration fails, moves the original file back to its original place.
#[cfg(any(not(any(target_os = "linux", target_os = "android")), test))]
fn safe_reflink(src: &PathAndMetadata, dest: &PathAndMetadata, log: &dyn Log) -> io::Result<()> {
    FsCommand::safe_remove(
        &dest.path,
        move |link| {
            copy_by_reflink(&src.path, link)?;
            Ok(())
        },
        log,
    )
}

// Dummy function so non-test cfg compiles
#[cfg(not(any(not(any(target_os = "linux", target_os = "android")), test)))]
fn safe_reflink(_src: &PathAndMetadata, _dest: &PathAndMetadata, _log: &dyn Log) -> io::Result<()> {
    unreachable!()
}

#[cfg(not(test))]
pub const fn crosstest() -> bool {
    false
}

#[cfg(test)]
pub fn crosstest() -> bool {
    test::cfg::crosstest()
}

#[cfg(test)]
pub mod test {
    pub mod cfg {
        // Helpers to switch reflink implementations when running tests
        // and to ensure only one reflink test runs at a time.

        use std::sync::{Mutex, MutexGuard};

        use lazy_static::lazy_static;

        lazy_static! {
            pub static ref CROSSTEST: Mutex<bool> = Mutex::new(false);
            pub static ref SEQUENTIAL_REFLINK_TESTS: Mutex<()> = Mutex::default();
        }

        pub struct CrossTest<'a>(#[allow(dead_code)] MutexGuard<'a, ()>);
        impl<'a> CrossTest<'a> {
            pub fn new(crosstest: bool) -> CrossTest<'a> {
                let x = CrossTest(SEQUENTIAL_REFLINK_TESTS.lock().unwrap());
                *CROSSTEST.lock().unwrap() = crosstest;
                x
            }
        }

        impl Drop for CrossTest<'_> {
            fn drop(&mut self) {
                *CROSSTEST.lock().unwrap() = false;
            }
        }

        pub fn crosstest() -> bool {
            *CROSSTEST.lock().unwrap()
        }
    }

    use crate::log::StdLog;
    use std::fs::File;
    use std::sync::Arc;

    use crate::util::test::{cached_reflink_supported, read_file, with_dir, write_file};

    use super::*;
    use crate::file::{FileChunk, FileHash, FileLen, FilePos};
    use crate::hasher::FileHasher;
    use crate::hasher::HashFn;
    use crate::path::Path as FcPath;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    // Helper function to compute hash of a file using the project's hasher
    fn compute_file_hash(path: &std::path::Path) -> FileHash {
        let log = StdLog::new();
        let hasher = FileHasher::new(HashFn::Metro, None, &log);
        let fc_path = FcPath::from(path);
        let chunk = FileChunk::new(&fc_path, FilePos(0), FileLen::MAX);
        hasher.hash_file(&chunk, |_| {}).unwrap()
    }

    // Helper to generate large files with specified content
    fn create_large_file(path: &std::path::Path, size_mb: usize, pattern_char: char) {
        let chunk_size = 1024 * 1024; // 1MB chunks
        let mut file = File::create(path).unwrap();

        for i in 0..size_mb {
            // Create a chunk with a unique pattern that includes the chunk number
            let mut chunk = format!("CHUNK{:04}:{}", i, pattern_char);
            // Pad to fill the chunk size
            chunk.push_str(&pattern_char.to_string().repeat(chunk_size - chunk.len()));
            file.write_all(chunk.as_bytes()).unwrap();
        }
    }

    // Helper to check if FIDEDUPERANGE is supported on this system
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn fideduperange_supported() -> bool {
        if !cached_reflink_supported() {
            println!("[fideduperange_supported] cached_reflink_supported() returned false, skipping");
            return false;
        }

        // Use with_dir instead of tempfile::tempdir() to ensure the temporary files
        // are created on the same filesystem as the project (e.g. btrfs),
        // not on /tmp which may be a different filesystem (e.g. ext4/tmpfs).
        let mut supported = false;
        with_dir("dedupe/fideduperange_support_check", |root| {
            let source_path = root.join("source_test");
            let dest_path = root.join("dest_test");

            println!("[fideduperange_supported] Testing in directory: {:?}", root);
            println!("[fideduperange_supported] source_path: {:?}", source_path);
            println!("[fideduperange_supported] dest_path: {:?}", dest_path);

            // Create identical files (use large content to rule out inline extent issues)
            let content = "foo".repeat(10000000);
            write_file(&source_path, &content);
            write_file(&dest_path, &content);

            // Try FIDEDUPERANGE
            let test_log = StdLog::new();
            let result = reflink_overwrite_dedupe(&source_path, &dest_path, &test_log);

            supported = match &result {
                Err(e) => {
                    let raw = e.raw_os_error();
                    println!(
                        "[fideduperange_supported] FIDEDUPERANGE failed: error={}, raw_os_error={:?}, ENOTTY={}, EOPNOTSUPP={}",
                        e, raw, libc::ENOTTY, libc::EOPNOTSUPP
                    );
                    if raw == Some(libc::ENOTTY) || raw == Some(libc::EOPNOTSUPP) {
                        false
                    } else {
                        // Other errors - still treat as unsupported but log it
                        println!(
                            "[fideduperange_supported] Unexpected error, treating as unsupported"
                        );
                        false
                    }
                }
                Ok(()) => {
                    println!("[fideduperange_supported] FIDEDUPERANGE succeeded!");
                    true
                }
            };
        });
        supported
    }

    // Usually /dev/shm only exists on Linux.
    #[cfg(target_os = "linux")]
    fn test_reflink_command_fails_on_dev_shm_tmpfs() {
        // No `cached_reflink_supported()` check

        if !std::path::Path::new("/dev/shm").is_dir() {
            println!("  Notice: strange Linux without /dev/shm, can't test reflink failure");
            return;
        }

        let test_root = "/dev/shm/tmp.fclones.reflink.testfailure";

        // Usually /dev/shm is mounted as a tmpfs which does not support reflinking, so test there.
        with_dir(test_root, |root| {
            // Always clean up files in /dev/shm, even after failure
            struct CleanupGuard<'a>(&'a str);
            impl Drop for CleanupGuard<'_> {
                fn drop(&mut self) {
                    fs::remove_dir_all(self.0).unwrap();
                }
            }
            let _guard = CleanupGuard(test_root);

            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            write_file(&file_path_1, "foo");
            write_file(&file_path_2, "foo");

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                mode: Some(crate::config::ReflinkMode::Safe),
                queue: None,
            };

            assert!(
                cmd.execute(true, &log)
                    .unwrap_err()
                    .to_string()
                    .starts_with("Failed to deduplicate"),
                "Reflink did not fail on /dev/shm (tmpfs), or this mount now supports reflinking"
            );

            assert!(file_path_2.exists());
            assert_eq!(read_file(&file_path_2), "foo");
        })
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_reflink_command_failure() {
        {
            let _sequential = cfg::CrossTest::new(false);
            test_reflink_command_fails_on_dev_shm_tmpfs();
        }
        {
            let _sequential = cfg::CrossTest::new(true);
            test_reflink_command_fails_on_dev_shm_tmpfs();
        }
    }

    fn test_reflink_command_with_file_too_large(via_ioctl: bool) {
        if !cached_reflink_supported() {
            return;
        }

        with_dir("dedupe/reflink_too_large", |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            write_file(&file_path_1, "foo");
            write_file(&file_path_2, "too large");

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                // Use Fast mode because files have different sizes;
                // Safe mode (FIDEDUPERANGE) cannot handle different-sized files.
                mode: Some(crate::config::ReflinkMode::Fast),
                queue: None,
            };

            if via_ioctl {
                assert!(cmd
                    .execute(true, &log)
                    .unwrap_err()
                    .to_string()
                    .starts_with("Failed to deduplicate"));

                assert!(file_path_1.exists());
                assert!(file_path_2.exists());
                assert_eq!(read_file(&file_path_1), "foo");
                assert_eq!(read_file(&file_path_2), "too large");
            } else {
                cmd.execute(true, &log).unwrap();

                assert!(file_path_2.exists());
                assert_eq!(read_file(&file_path_2), "foo");
            }
        })
    }

    #[test]
    fn test_reflink_command_works_with_files_too_large_anyos() {
        let _sequential = cfg::CrossTest::new(true);
        test_reflink_command_with_file_too_large(false);
    }

    // This tests the reflink code path (using the reflink crate) usually not used on Linux.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_command_fails_with_files_too_large_using_ioctl_linux() {
        let _sequential = cfg::CrossTest::new(false);
        test_reflink_command_with_file_too_large(true);
    }

    fn test_reflink_command_fills_file_with_content() {
        if !cached_reflink_supported() {
            return;
        }
        with_dir("dedupe/reflink_test", |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            write_file(&file_path_1, "foo");
            write_file(&file_path_2, "f");

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                // Use Fast mode because files have different sizes;
                // Safe mode (FIDEDUPERANGE) cannot handle different-sized files.
                mode: Some(crate::config::ReflinkMode::Fast),
                queue: None,
            };
            cmd.execute(true, &log).unwrap();

            assert!(file_path_1.exists());
            assert!(file_path_2.exists());
            assert_eq!(read_file(&file_path_2), "foo");
        })
    }

    #[test]
    fn test_reflink_command_fills_file_with_content_anyos() {
        let _sequential = cfg::CrossTest::new(false);
        test_reflink_command_fills_file_with_content();
    }

    // This tests the reflink code path (using the reflink crate) usually not used on Linux.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_command_fills_file_with_content_not_ioctl_linux() {
        let _sequential = cfg::CrossTest::new(true);
        test_reflink_command_fills_file_with_content();
    }

    // Test that safe mode (FIDEDUPERANGE) fails when files have different sizes.
    // `link_content` is the content of the link file (target is always "foo").
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_fails_with_different_sizes(dir_name: &str, link_content: &str) {
        if !fideduperange_supported() {
            println!("Skipping test: FIDEDUPERANGE not supported on this system");
            return;
        }

        with_dir(dir_name, |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            write_file(&file_path_1, "foo");
            write_file(&file_path_2, link_content);

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                mode: Some(crate::config::ReflinkMode::Safe),
                queue: None,
            };

            // Safe mode should fail because files have different sizes
            let result = cmd.execute(true, &log);
            assert!(
                result.is_err(),
                "Safe mode should fail when files have different sizes (link='{link_content}')"
            );
            assert!(
                result.unwrap_err().to_string().starts_with("Failed to deduplicate"),
                "Error message should indicate deduplication failure"
            );

            // Both files should remain unchanged
            assert!(file_path_1.exists());
            assert!(file_path_2.exists());
            assert_eq!(read_file(&file_path_1), "foo");
            assert_eq!(read_file(&file_path_2), link_content);
        })
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_fails_when_link_smaller_than_target() {
        let _sequential = cfg::CrossTest::new(false);
        test_reflink_safe_fails_with_different_sizes("dedupe/reflink_safe_link_smaller", "f");
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_fails_when_link_larger_than_target() {
        let _sequential = cfg::CrossTest::new(false);
        test_reflink_safe_fails_with_different_sizes("dedupe/reflink_safe_link_larger", "too large");
    }

    // Test that safe mode (FIDEDUPERANGE) succeeds when files have the same size and identical content.
    // `repeat_count` controls how many times "foo" or "fool" is repeated to form the file content.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_succeeds_with_identical_content(dir_name: &str, content: &str) {
        if !fideduperange_supported() {
            println!("Skipping test: FIDEDUPERANGE not supported on this system");
            return;
        }

        with_dir(dir_name, |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            // Create two files with identical content
            write_file(&file_path_1, content);
            write_file(&file_path_2, content);

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                mode: Some(crate::config::ReflinkMode::Safe),
                queue: None,
            };

            // Safe mode should succeed with identical files
            let result = cmd.execute(true, &log);
            assert!(
                result.is_ok(),
                "Safe mode should succeed when files have identical content (size={} bytes): {:?}",
                content.len(),
                result.unwrap_err()
            );

            // Both files should still exist with the same content
            assert!(file_path_1.exists());
            assert!(file_path_2.exists());
            assert_eq!(read_file(&file_path_1), content);
            assert_eq!(read_file(&file_path_2), content);
        })
    }

    // Macro to generate parameterized safe mode tests with identical content
    macro_rules! test_reflink_safe_identical {
        ($name:ident, $pattern:expr, $count:expr) => {
            #[test]
            #[cfg(any(target_os = "linux", target_os = "android"))]
            fn $name() {
                let _sequential = cfg::CrossTest::new(false);
                let content = $pattern.repeat($count);
                let dir_name = format!("dedupe/reflink_safe_{}_{}", $count, $pattern);
                test_reflink_safe_succeeds_with_identical_content(&dir_name, &content);
            }
        };
    }

    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_1_foo, "foo", 1);          // 3 bytes
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_10_foo, "foo", 10);         // 30 bytes
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_100_foo, "foo", 100);       // 300 bytes
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_1000_foo, "foo", 1000);     // 3000 bytes
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_100000_foo, "foo", 100000);   // 300000 bytes
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_1000000_foo, "foo", 1000000);   // 3000000 bytes (~3MB)
    test_reflink_safe_identical!(test_reflink_safe_succeeds_with_10000000_foo, "foo", 10000000); // 30000000 bytes (~30MB)

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_rejects_with_appended_content(dir_name: &str, content: &str) {
        if !fideduperange_supported() {
            println!("Skipping test: FIDEDUPERANGE not supported on this system");
            return;
        }

        let content_with_bar = format!("{}bar", content);

        with_dir(dir_name, |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");

            // Source has original content, dest has content + "bar" appended at end
            write_file(&file_path_1, content);
            write_file(&file_path_2, &content_with_bar);

            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                mode: Some(crate::config::ReflinkMode::Safe),
                queue: None,
            };

            // Safe mode should fail because files have different content (and size)
            let result = cmd.execute(true, &log);
            assert!(
                result.is_err(),
                "Safe mode should reject files when dest has 'bar' appended (source={} bytes, dest={} bytes)",
                content.len(),
                content_with_bar.len()
            );
            assert!(
                result.unwrap_err().to_string().starts_with("Failed to deduplicate"),
                "Error message should indicate deduplication failure"
            );

            // Both files should remain unchanged
            assert!(file_path_1.exists());
            assert!(file_path_2.exists());
            assert_eq!(read_file(&file_path_1), content);
            assert_eq!(read_file(&file_path_2), content_with_bar);
        })
    }

    // Macro to generate parameterized safe mode rejection tests with "bar" appended at end
    macro_rules! test_reflink_safe_rejects {
        ($name:ident, $pattern:expr, $count:expr) => {
            #[test]
            #[cfg(any(target_os = "linux", target_os = "android"))]
            fn $name() {
                let _sequential = cfg::CrossTest::new(false);
                let content = $pattern.repeat($count);
                let dir_name = format!("dedupe/reflink_safe_rejects_{}_{}", $count, $pattern);
                test_reflink_safe_rejects_with_appended_content(&dir_name, &content);
            }
        };
    }

    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_1_foo, "foo", 1);            // 3 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_10_foo, "foo", 10);           // 30 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_100_foo, "foo", 100);         // 300 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_1000_foo, "foo", 1000);       // 3000 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_100000_foo, "foo", 100000);   // 300000 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_1000000_foo, "foo", 1000000);   // 3000000 bytes + "bar"
    test_reflink_safe_rejects!(test_reflink_safe_rejects_with_10000000_foo, "foo", 10000000); // 30000000 bytes + "bar"

    // Test that safe mode (FIDEDUPERANGE) does NOT modify the mtime or create time (birth time)
    // of the destination file.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_reflink_safe_preserves_timestamps_impl(dir_name: &str, content: &str) {
        use filetime::FileTime;
        use std::thread;
        use std::time::Duration;

        if !fideduperange_supported() {
            println!("Skipping test: FIDEDUPERANGE not supported on this system");
            return;
        }

        with_dir(dir_name, |root| {
            let log = StdLog::new();
            let file_path_1 = root.join("file_1");
            let file_path_2 = root.join("file_2");
            let file_path_3 = root.join("file_3");

            // Create source file
            write_file(&file_path_1, content);

            // Create dest file and record its timestamps (old2)
            write_file(&file_path_2, content);
            let meta_old2 = fs::metadata(&file_path_2).unwrap();
            let mtime_old2 = FileTime::from_last_modification_time(&meta_old2);
            let ctime_old2 = meta_old2.created().ok();

            // Sleep 0.1s to ensure any new file write would have a later timestamp
            thread::sleep(Duration::from_millis(100));

            // Write a new file_path_3 and read its mtime (mtime3)
            write_file(&file_path_3, content);
            let meta3 = fs::metadata(&file_path_3).unwrap();
            let mtime3 = FileTime::from_last_modification_time(&meta3);

            // Assert mtime3 is later than old2 (proves the clock has advanced)
            assert!(
                mtime3 > mtime_old2,
                "mtime of file_3 ({:?}) should be later than mtime of file_2 ({:?}), proving clock advancement",
                mtime3, mtime_old2
            );

            // Assert create time of file_3 is later than file_2 if supported
            let ctime3 = meta3.created().ok();
            if let (Some(ct_old2), Some(ct3)) = (ctime_old2, ctime3) {
                assert!(
                    ct3 > ct_old2,
                    "create time of file_3 ({:?}) should be later than create time of file_2 ({:?}), proving clock advancement",
                    ct3, ct_old2
                );
            } else {
                println!(
                    "Note: create time (birth time) not available on this filesystem, skipping create time clock advancement check"
                );
            }

            // Now perform reflink from file_path_1 to file_path_2
            let file_1 = PathAndMetadata::new(FcPath::from(&file_path_1)).unwrap();
            let file_2 = PathAndMetadata::new(FcPath::from(&file_path_2)).unwrap();
            let cmd = FsCommand::RefLink {
                target: Arc::new(file_1),
                link: file_2,
                mode: Some(crate::config::ReflinkMode::Safe),
                queue: None,
            };

            let result = cmd.execute(true, &log);
            assert!(
                result.is_ok(),
                "Safe mode should succeed with identical content (size={} bytes): {:?}",
                content.len(),
                result.unwrap_err()
            );

            // Read file_path_2's mtime after dedup (new2), assert old2 == new2
            let meta_new2 = fs::metadata(&file_path_2).unwrap();
            let mtime_new2 = FileTime::from_last_modification_time(&meta_new2);

            assert_eq!(
                mtime_old2, mtime_new2,
                "Safe mode should NOT modify mtime of dest file (old2={:?}, new2={:?})",
                mtime_old2, mtime_new2
            );

            // Verify create time is also preserved after dedup
            let ctime_new2 = meta_new2.created().ok();
            if let (Some(ct_old2), Some(ct_new2)) = (ctime_old2, ctime_new2) {
                assert_eq!(
                    ct_old2, ct_new2,
                    "Safe mode should NOT modify create time of dest file (old2={:?}, new2={:?})",
                    ct_old2, ct_new2
                );
            } else {
                println!(
                    "Note: create time (birth time) not available on this filesystem, skipping create time preservation check"
                );
            }

            // Also verify content is still correct
            assert_eq!(read_file(&file_path_1), content);
            assert_eq!(read_file(&file_path_2), content);
        })
    }

    macro_rules! test_reflink_safe_preserves_timestamps {
        ($name:ident, $pattern:expr, $count:expr) => {
            #[test]
            #[cfg(any(target_os = "linux", target_os = "android"))]
            fn $name() {
                let _sequential = cfg::CrossTest::new(false);
                let content = $pattern.repeat($count);
                let dir_name = format!("dedupe/reflink_safe_timestamps_{}_{}", $count, $pattern);
                test_reflink_safe_preserves_timestamps_impl(&dir_name, &content);
            }
        };
    }

    test_reflink_safe_preserves_timestamps!(test_reflink_safe_preserves_timestamps_with_1000000_foo, "foo", 1000000);   // 3000000 bytes (~3MB)

}
