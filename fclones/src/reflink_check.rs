//! Check whether two files share physical extents (reflink detection) via FIEMAP ioctl.
//!
//! This module is Linux-only.

use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;

// ---------------------------------------------------------------------------
// FIEMAP ioctl definitions (from <linux/fiemap.h> and <linux/fs.h>)
// ---------------------------------------------------------------------------

const FS_IOC_FIEMAP: libc::c_ulong = 0xC020660B; // _IOWR('f', 11, struct fiemap)
const FIEMAP_FLAG_SYNC: u32 = 0x0001;
const FIEMAP_EXTENT_DATA_INLINE: u32 = 0x0200;
const FIEMAP_EXTENT_SHARED: u32 = 0x2000;
const FIEMAP_EXTENT_LAST: u32 = 0x0001;

/// Mirrors `struct fiemap` (variable-length header).
#[repr(C)]
struct Fiemap {
    fm_start: u64,
    fm_length: u64,
    fm_flags: u32,
    fm_mapped_extents: u32,
    fm_extent_count: u32,
    fm_reserved: u32,
    // followed by fm_extents[fm_extent_count]
}

/// Mirrors `struct fiemap_extent`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FiemapExtent {
    fe_logical: u64,
    fe_physical: u64,
    fe_length: u64,
    fe_reserved64: [u64; 2],
    fe_flags: u32,
    fe_reserved: [u32; 3],
}

impl fmt::Display for FiemapExtent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tags = String::new();
        if self.fe_flags & FIEMAP_EXTENT_DATA_INLINE != 0 {
            tags.push_str(" INLINE");
        }
        if self.fe_flags & FIEMAP_EXTENT_SHARED != 0 {
            tags.push_str(" SHARED");
        }
        if self.fe_flags & FIEMAP_EXTENT_LAST != 0 {
            tags.push_str(" LAST");
        }
        write!(
            f,
            "log={:#x} phys={:#x} len={:#x} flags={:#x}{}",
            self.fe_logical, self.fe_physical, self.fe_length, self.fe_flags, tags
        )
    }
}

// ---------------------------------------------------------------------------
// get_extents: fetch all FIEMAP extents of an open file descriptor
// ---------------------------------------------------------------------------

/// Maximum number of extents to fetch per ioctl call.
/// Uses a fixed-size buffer (~64KB) similar to filefrag, avoiding large
/// allocations for heavily fragmented files and eliminating the TOCTOU
/// race of the previous two-phase approach.
const EXTENT_BATCH_SIZE: u32 = 768;

fn get_extents(fd: i32) -> io::Result<Vec<FiemapExtent>> {
    let buf_size = std::mem::size_of::<Fiemap>()
        + EXTENT_BATCH_SIZE as usize * std::mem::size_of::<FiemapExtent>();
    let mut buf = vec![0u8; buf_size];
    let mut all_extents: Vec<FiemapExtent> = Vec::new();
    let mut fm_start: u64 = 0;

    loop {
        // Zero out the buffer for safety
        buf.iter_mut().for_each(|b| *b = 0);

        // Fill header
        let fm = unsafe { &mut *(buf.as_mut_ptr() as *mut Fiemap) };
        fm.fm_start = fm_start;
        fm.fm_length = u64::MAX;
        fm.fm_flags = FIEMAP_FLAG_SYNC;
        fm.fm_extent_count = EXTENT_BATCH_SIZE;

        let ret = unsafe { libc::ioctl(fd, FS_IOC_FIEMAP, buf.as_mut_ptr()) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        let fm = unsafe { &*(buf.as_ptr() as *const Fiemap) };
        let mapped = fm.fm_mapped_extents as usize;
        if mapped == 0 {
            break;
        }

        // Read extents from the trailing part of buf
        let extents_ptr = unsafe {
            buf.as_ptr()
                .add(std::mem::size_of::<Fiemap>())
                .cast::<FiemapExtent>()
        };

        let mut done = false;
        for i in 0..mapped {
            let extent = unsafe { *extents_ptr.add(i) };
            // Advance fm_start past this extent for the next iteration
            fm_start = extent.fe_logical + extent.fe_length;
            all_extents.push(extent);

            if extent.fe_flags & FIEMAP_EXTENT_LAST != 0 {
                done = true;
                break;
            }
        }

        if done {
            break;
        }
    }

    Ok(all_extents)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a reflink check between two files.
#[derive(Debug)]
pub enum CheckResult {
    /// Files share all extents (reflinked).
    Reflinked(usize),
    /// Not reflinked, with a human-readable reason.
    NotReflinked(String),
}

/// Check if two files share physical extents (reflink detection).
///
/// Opens both files, reads their FIEMAP extents, and compares them.
/// Returns `CheckResult::Reflinked(n)` if all `n` extents match,
/// or `CheckResult::NotReflinked(reason)` otherwise.
///
/// # Accuracy
///
/// This check is **one-directional safe**:
/// - **No false negatives**: if two files are truly reflinked, this function
///   will always return `Reflinked`. A reflinked pair shares the same physical
///   extents by definition, so the FIEMAP comparison is guaranteed to match.
/// - **Theoretically possible false positives**: a non-reflinked pair could be
///   reported as `Reflinked` if every extent happens to have identical
///   `fe_logical`, `fe_physical`, and `fe_length` values by coincidence. In
///   practice this is near-impossible because the filesystem allocator will not
///   assign identical physical regions to two independently written files that
///   coexist at the same time.
pub fn fast_check_reflinked(path1: &str, path2: &str, mute: bool) -> io::Result<CheckResult> {
    let f1 = File::open(path1)?;
    let f2 = File::open(path2)?;

    let ext1 = get_extents(f1.as_raw_fd())?;
    let ext2 = get_extents(f2.as_raw_fd())?;

    // Print extent info (suppressed when mute is true)
    if !mute {
        println!("extents: file1={}  file2={}", ext1.len(), ext2.len());
        for (i, e) in ext1.iter().enumerate() {
            println!("  f1[{}] {}", i, e);
        }
        for (i, e) in ext2.iter().enumerate() {
            println!("  f2[{}] {}", i, e);
        }
    }

    // Inline extents cannot be reflinked
    for (i, e) in ext1.iter().enumerate() {
        if e.fe_flags & FIEMAP_EXTENT_DATA_INLINE != 0 {
            return Ok(CheckResult::NotReflinked(format!(
                "file1 extent[{}] is INLINE, reflink not applicable",
                i
            )));
        }
    }
    for (i, e) in ext2.iter().enumerate() {
        if e.fe_flags & FIEMAP_EXTENT_DATA_INLINE != 0 {
            return Ok(CheckResult::NotReflinked(format!(
                "file2 extent[{}] is INLINE, reflink not applicable",
                i
            )));
        }
    }

    // Extent count must match
    if ext1.len() != ext2.len() {
        return Ok(CheckResult::NotReflinked(format!(
            "extent count differs: {} vs {}",
            ext1.len(),
            ext2.len()
        )));
    }

    // Every extent must occupy the same logical offset and physical region
    let mut shared_count = 0usize;
    for i in 0..ext1.len() {
        let a = &ext1[i];
        let b = &ext2[i];
        if a.fe_logical != b.fe_logical {
            return Ok(CheckResult::NotReflinked(format!(
                "extent[{}]: logical offset differs: {:#x} vs {:#x}",
                i, a.fe_logical, b.fe_logical
            )));
        }
        if a.fe_physical != b.fe_physical || a.fe_length != b.fe_length {
            return Ok(CheckResult::NotReflinked(format!(
                "extent[{}]: {:#x}/{:#x} vs {:#x}/{:#x}",
                i, a.fe_physical, a.fe_length, b.fe_physical, b.fe_length
            )));
        }
        if a.fe_flags & FIEMAP_EXTENT_SHARED != 0 {
            shared_count += 1;
        }
    }

    // If the kernel reports SHARED flags, verify consistency:
    // all extents should be marked SHARED for a fully reflinked pair.
    if shared_count > 0 && shared_count != ext1.len() {
        return Ok(CheckResult::NotReflinked(format!(
            "only {}/{} extents have FIEMAP_EXTENT_SHARED flag set",
            shared_count,
            ext1.len()
        )));
    }

    Ok(CheckResult::Reflinked(ext1.len()))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::test::{cached_reflink_supported, with_dir, write_file};

    /// Helper: create a reflink copy of `src` at `dest` using FICLONE ioctl.
    /// Returns `Ok(())` on success, or an `io::Error` on failure.
    fn create_reflink(src: &std::path::Path, dest: &std::path::Path) -> io::Result<()> {
        use nix::request_code_write;
        use std::os::unix::prelude::AsRawFd;

        let src_file = std::fs::File::open(src)?;
        let dest_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(dest)?;

        const FICLONE_TYPE: u8 = 0x94;
        const FICLONE_NR: u8 = 9;
        const FICLONE_SIZE: usize = std::mem::size_of::<libc::c_int>();

        let ret = unsafe {
            libc::ioctl(
                dest_file.as_raw_fd(),
                request_code_write!(FICLONE_TYPE, FICLONE_NR, FICLONE_SIZE),
                src_file.as_raw_fd(),
            )
        };

        if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Positive: two files sharing extents via reflink should be detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_detects_reflinked_files() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/detect_reflinked", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            // Write a large enough file to avoid inline extents
            let content = "foo".repeat(100_000); // 300 KB
            write_file(&file1, &content);

            // Create a reflink copy
            create_reflink(&file1, &file2)
                .expect("FICLONE should succeed on a reflink-capable filesystem");

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::Reflinked(n) => {
                    assert!(n > 0, "Expected at least one shared extent after read");                    println!("OK: files share {} extent(s)", n);
                }
                CheckResult::NotReflinked(reason) => {
                    panic!("Expected Reflinked, got NotReflinked: {}", reason);
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Negative: two independently written files with identical content
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_independent_identical_files() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/independent_identical", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            let content = "bar".repeat(100_000);
            write_file(&file1, &content);
            write_file(&file2, &content);

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::NotReflinked(reason) => {
                    println!("OK: independent files are not reflinked ({})", reason);
                }
                CheckResult::Reflinked(_) => {
                    // Some CoW filesystems may automatically deduplicate identical
                    // blocks, so this is acceptable but unexpected.
                    println!("WARN: filesystem auto-deduplicated identical blocks");
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Negative: two files with different content
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_different_content() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/different_content", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            let content1 = "aaa".repeat(100_000);
            let content2 = "bbb".repeat(100_000);
            write_file(&file1, &content1);
            write_file(&file2, &content2);

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::NotReflinked(reason) => {
                    println!("OK: different files are not reflinked ({})", reason);
                }
                CheckResult::Reflinked(_) => {
                    panic!("Different files should never be detected as reflinked");
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Negative: files with different sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_different_sizes() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/different_sizes", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            let content1 = "x".repeat(100_000);
            let content2 = "x".repeat(200_000);
            write_file(&file1, &content1);
            write_file(&file2, &content2);

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::NotReflinked(reason) => {
                    println!("OK: different-sized files are not reflinked ({})", reason);
                }
                CheckResult::Reflinked(_) => {
                    panic!("Files with different sizes should not be detected as reflinked");
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Error: non-existent file
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_nonexistent_file() {
        let result = fast_check_reflinked("/nonexistent/path/a", "/nonexistent/path/b", false);
        assert!(result.is_err(), "Non-existent files should produce IO error");
    }

    // -----------------------------------------------------------------------
    // Edge case: empty files
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_empty_files() {
        with_dir("fast_reflink_check/empty_files", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            write_file(&file1, "");
            write_file(&file2, "");

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            // Empty files have no extents; they are trivially "reflinked" (0 extents match)
            match result {
                CheckResult::Reflinked(n) => {
                    assert_eq!(n, 0, "Empty files should have 0 shared extents");
                }
                CheckResult::NotReflinked(reason) => {
                    println!("OK: empty files considered not reflinked ({})", reason);
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Positive: reflinked file still detected after reading content
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_after_read() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/after_read", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            let content = "hello".repeat(100_000);
            write_file(&file1, &content);
            create_reflink(&file1, &file2)
                .expect("FICLONE should succeed");

            // Read both files fully to ensure OS caching doesn't affect FIEMAP
            let data1 = std::fs::read_to_string(&file1).unwrap();
            let data2 = std::fs::read_to_string(&file2).unwrap();
            assert_eq!(data1, data2, "Reflinked files should have identical content");

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::Reflinked(n) => {
                    assert!(n > 0, "Expected at least one shared extent after read");
                }
                CheckResult::NotReflinked(reason) => {
                    panic!("Reflinked files should still share extents after read: {}", reason);
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Negative: reflink broken after overwrite (CoW triggers new extent)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_broken_after_overwrite() {
        if !cached_reflink_supported() {
            println!("Skipping: filesystem does not support reflinks");
            return;
        }

        with_dir("fast_reflink_check/broken_after_overwrite", |root| {
            let file1 = root.join("file1");
            let file2 = root.join("file2");

            let content = "data".repeat(100_000);
            write_file(&file1, &content);
            create_reflink(&file1, &file2)
                .expect("FICLONE should succeed");

            // Overwrite file2 entirely with different content to break CoW sharing
            let new_content = "NEW!".repeat(100_000);
            write_file(&file2, &new_content);

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file2.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::NotReflinked(reason) => {
                    println!("OK: overwritten reflink no longer shares extents ({})", reason);
                }
                CheckResult::Reflinked(_) => {
                    panic!("After full overwrite, files should no longer share extents");
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Positive: self-check — a file should share extents with itself
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_check_reflinked_same_file() {
        with_dir("fast_reflink_check/same_file", |root| {
            let file1 = root.join("file1");
            let content = "self".repeat(100_000);
            write_file(&file1, &content);

            let result = fast_check_reflinked(
                file1.to_str().unwrap(),
                file1.to_str().unwrap(),
                false,
            )
            .expect("fast_check_reflinked should not return an IO error");

            match result {
                CheckResult::Reflinked(n) => {
                    assert!(n > 0, "A file should share extents with itself");
                }
                CheckResult::NotReflinked(reason) => {
                    panic!("A file compared with itself should be Reflinked, got: {}", reason);
                }
            }
        });
    }
}
