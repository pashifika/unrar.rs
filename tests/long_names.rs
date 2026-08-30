//! An entry name longer than the DLL's fixed 1024-wchar field survives intact.
//!
//! `HeaderDataEx::FileNameW` is a `[wchar_t; 1024]` and `dll.cpp` fills it with `wcsncpyz(..,
//! ASIZE(D->FileNameW))`, so a longer name used to come back cut with nothing to say it had
//! been. A caller keying on the extension then saw a name with no extension — a file silently
//! missing from the archive it was listing.
//!
//! `data/long-name.rar` holds two entries, one with a 1265-character name and one with a
//! 12-character name. Independently confirmed: `bsdtar -tf` reports name lengths 1265 and 12.

use std::path::PathBuf;

fn names(path: &str) -> Vec<PathBuf> {
    let mut archive = unrar_ng::Archive::new(path)
        .open_for_listing()
        .expect("open for listing");
    let mut found = Vec::new();
    while let Some(entry) = archive.next() {
        found.push(entry.expect("header").filename);
    }
    found
}

#[test]
fn a_name_longer_than_the_fixed_field_is_not_truncated() {
    let found = names("data/long-name.rar");
    let lengths: Vec<_> = found
        .iter()
        .map(|p| p.to_string_lossy().chars().count())
        .collect();
    assert_eq!(
        lengths,
        vec![1265, 12],
        "the long name was cut; 1023 is the tell"
    );
}

#[test]
fn the_long_name_keeps_its_extension() {
    let found = names("data/long-name.rar");
    let first = found[0].to_string_lossy().into_owned();
    assert!(
        first.ends_with(".jpg"),
        "the extension is what a caller filters on, and it is at the end that was cut"
    );
}

/// The processing cursor reads headers through the same path, so it must agree.
#[test]
fn the_processing_cursor_sees_the_same_names() {
    let mut archive = unrar_ng::Archive::new("data/long-name.rar")
        .open_for_processing()
        .expect("open");
    let mut found = Vec::new();
    while let Some(header) = archive.read_header().expect("header") {
        found.push(header.entry().filename.to_string_lossy().chars().count());
        archive = header.skip().expect("skip");
    }
    assert_eq!(found, vec![1265, 12]);
}
