//! An open archive can be handed to another thread.
//!
//! A caller that reads archives on a worker thread has to construct the archive somewhere and
//! move it there. Without `Send` that is impossible, whatever the runtime behaviour would have
//! been.

use unrar_ng::{CursorBeforeHeader, OpenArchive, Process};

const VERSION: &[u8] = b"unrar-0.4.0";

#[test]
fn an_open_archive_is_send() {
    const fn assert_send<T: Send>() {}
    assert_send::<OpenArchive<Process, CursorBeforeHeader>>();
}

/// Not only the type check: the archive is opened on this thread, moved, and driven to
/// completion on another. A `Send` claim that was never exercised would be a claim.
#[test]
fn an_archive_opened_here_reads_there() {
    let archive = unrar_ng::Archive::new("data/version.rar")
        .open_for_processing()
        .expect("open");

    let bytes = std::thread::spawn(move || {
        let (bytes, _archive) = archive
            .read_header()
            .expect("read header")
            .expect("an entry")
            .read()
            .expect("read");
        bytes
    })
    .join()
    .expect("the reader thread did not panic");

    assert_eq!(bytes, VERSION);
}

/// Two archives, two threads, at the same time. libunrar keeps per-archive state behind the
/// handle, so this must work; the global `ErrHandler` is what the `Send` impl's safety note
/// is about, and this is the shape that would expose it.
#[test]
fn two_archives_read_on_two_threads() {
    let handles: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                let archive = unrar_ng::Archive::new("data/version.rar")
                    .open_for_processing()
                    .expect("open");
                let (bytes, _archive) = archive
                    .read_header()
                    .expect("read header")
                    .expect("an entry")
                    .read()
                    .expect("read");
                bytes
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().expect("no panic"), VERSION);
    }
}
