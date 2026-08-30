use super::error::*;
use super::*;
use std::fmt;
use std::os::raw::{c_int, c_uint};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard, PoisonError};

bitflags::bitflags! {
    #[derive(Debug, Default)]
    struct ArchiveFlags: u32 {
        const VOLUME = native::ROADF_VOLUME;
        const COMMENT = native::ROADF_COMMENT;
        const LOCK = native::ROADF_LOCK;
        const SOLID = native::ROADF_SOLID;
        const NEW_NUMBERING = native::ROADF_NEWNUMBERING;
        const SIGNED = native::ROADF_SIGNED;
        const RECOVERY = native::ROADF_RECOVERY;
        const ENC_HEADERS = native::ROADF_ENCHEADERS;
        const FIRST_VOLUME = native::ROADF_FIRSTVOLUME;
    }
}

/// Volume information on the file that was *initially* opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeInfo {
    /// the *initially* opened file is a single-part archive
    None,
    /// the *initially* opened file is the first volume in a multipart archive
    First,
    /// the *initially* opened file is any volume but the first in a multipart archive
    Subsequent,
}

/// Extraction progress event for callbacks during batch extraction.
///
/// This enum is used with [`OpenArchive::extract_all_with_callback`] to receive
/// notifications about extraction progress.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExtractEvent {
    /// File extraction is starting.
    Start {
        /// The filename being extracted (relative path within the archive)
        filename: PathBuf,
        /// The uncompressed size of the file in bytes
        size: u64,
    },
    /// File extraction completed successfully.
    Ok {
        /// The filename that was extracted
        filename: PathBuf,
        /// The uncompressed size of the file in bytes, carried over
        /// from the matching `Start` event so callers do not have to
        /// stash it themselves to log progress.
        size: u64,
    },
    /// File extraction failed.
    Err {
        /// The filename that failed to extract
        filename: PathBuf,
        /// The error code from the extraction
        error_code: i32,
    },
    /// The archive requires a dictionary larger than the build-time limit.
    ///
    /// Surfaced from the upstream `UCM_LARGEDICT` callback. Returning
    /// `true` permits the DLL to proceed; returning `false` lets the DLL
    /// fail the operation, which the caller then observes as
    /// `Err(UnrarError { code:` [`Code::LargeDict`](crate::error::Code::LargeDict)`, when: When::Process })`.
    LargeDictWarning {
        /// The dictionary size required by the archive, in kilobytes.
        dict_size_kb: u64,
        /// The maximum dictionary size this build supports, in kilobytes.
        max_dict_size_kb: u64,
    },
}

/// Outcome of [`OpenArchive::extract_all_with_callback`].
///
/// The DLL maps a user-initiated cancel (the callback returning `false`
/// from `Start`/`Ok`/`Err`) to `ERAR_SUCCESS`, so without this status
/// the caller cannot tell whether the archive finished or was stopped
/// early. Pattern-match to distinguish the two — `Completed` means the
/// DLL exhausted the archive, `Cancelled` means the callback aborted
/// the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExtractStatus {
    /// Extraction ran to the end of the archive.
    Completed,
    /// The user callback returned `false` and aborted extraction early.
    Cancelled,
}

/// Serialises every call into libunrar, which is what makes [`Handle`]'s `Send` sound.
///
/// libunrar is not thread-safe, and the unsafety is not confined to one archive's state. A
/// thread-sanitiser harness that opened two archives, moved each to its own thread and drove
/// them at once reported races in
/// `Archive::ConvertAttributes`'s function-local `mask` (`arcread.cpp`, which wraps the
/// process-wide `umask`), in `Unpack29::DDecode`'s lazily built tables, and in the global
/// `ErrHandler::SetErrorCode` reached from `rdwrfn.cpp` on the abort path. Upstream's
/// `vendor/patches/0002` removed `ErrHandler.GetErrorCode()` from the DLL's *return* paths,
/// which is narrower than thread safety and does not cover any of these.
///
/// So `Send` cannot rest on "each archive owns its own state": it does not. It rests on this
/// lock, which is held across each complete transaction rather than each call, because the
/// callback registered by `RARSetCallback` belongs to the operation that follows it and a
/// second thread registering its own in between would redirect ours.
///
/// # A sink or callback must not re-enter this crate
///
/// The lock is not reentrant and user code runs under it — [`DataSink::write_chunk`] and the
/// [`ExtractEvent`] callback both do. Opening or reading another archive from inside one
/// deadlocks. Recorded rather than defended against, because the alternative is a reentrant
/// lock that would let a second archive's calls interleave with the first's, which is the
/// thing this exists to prevent.
static DLL: Mutex<()> = Mutex::new(());

/// Takes the lock for one libunrar transaction.
///
/// Poisoning is ignored: the mutex guards no Rust data, only the right to be inside the
/// library, and a panicking caller leaves libunrar no more broken than it left itself.
fn serialised() -> MutexGuard<'static, ()> {
    DLL.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Debug)]
struct Handle(NonNull<native::Handle>);

impl Drop for Handle {
    fn drop(&mut self) {
        let _guard = serialised();
        unsafe { native::RARCloseArchive(self.0.as_ptr() as *const _) };
    }
}

// SAFETY: the handle owns libunrar's per-archive `DataSet`, and ownership here is exclusive —
// `Handle` is neither `Clone` nor `Copy` and closes the archive on drop, so exactly one
// `OpenArchive` ever refers to a given `DataSet` and every call goes through `&mut self` or
// consumes `self`. Moving that ownership to another thread hands the archive over rather than
// sharing it.
//
// That alone would NOT be enough, and an earlier version of this comment wrongly said it was.
// libunrar has process-wide mutable state that two archives on two threads both reach; see
// [`DLL`] for the specific races a thread sanitiser found. Soundness rests on that lock
// serialising every transaction, not on per-archive ownership.
//
// Deliberately `Send` and not `Sync`: sharing one archive between threads would still be
// wrong, because the type-state cursor is not atomic.
unsafe impl Send for Handle {}

/// An open RAR archive that can be read or processed.
///
/// See the [OpenArchive chapter](index.html#openarchive) for more information.
#[derive(Debug)]
pub struct OpenArchive<M: OpenMode, C: Cursor> {
    handle: Handle,
    flags: ArchiveFlags,
    damaged: bool,
    extra: C,
    marker: std::marker::PhantomData<M>,
}
/// Per-call state the C callback writes into.
///
/// The third slot records that *we* aborted the read, because the DLL cannot tell us:
/// `RARX_USERBREAK` has no `RarErrorToDll` case and surfaces as `ERAR_UNKNOWN`.
type Userdata<T> = (T, Option<widestring::WideCString>, bool);

mod private {
    use super::native;
    pub trait Sealed {}
    impl Sealed for super::CursorBeforeHeader {}
    impl Sealed for super::CursorBeforeFile {}
    impl Sealed for super::List {}
    impl Sealed for super::ListSplit {}
    impl Sealed for super::Process {}

    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Operation {
        Skip = native::RAR_SKIP,
        Test = native::RAR_TEST,
        Extract = native::RAR_EXTRACT,
    }

    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OpenModeValue {
        Extract = native::RAR_OM_EXTRACT,
        List = native::RAR_OM_LIST,
        ListIncSplit = native::RAR_OM_LIST_INCSPLIT,
    }
}

/// Type parameter for OpenArchive denoting a `read_header` operation must follow next.
///
/// See the chapter [OpenArchive: Cursors](index.html#openarchive-cursors) for more information.
#[derive(Debug)]
pub struct CursorBeforeHeader;
/// Type parameter for OpenArchive denoting a `process_file` operation must follow next.
///
/// See the chapter [OpenArchive: Cursors](index.html#openarchive-cursors) for more information.
#[derive(Debug)]
pub struct CursorBeforeFile {
    header: FileHeader,
}

/// The Cursor trait enables archives to keep track of their state.
///
/// See the chapter [OpenArchive: Cursors](index.html#openarchive-cursors) for more information.
pub trait Cursor: private::Sealed {}
impl Cursor for CursorBeforeHeader {}
impl Cursor for CursorBeforeFile {}

/// An OpenMode for processing RAR archive entries.
///
/// Process allows more sophisticated operations in the `ProcessFile` step.
#[derive(Debug)]
pub struct Process;
#[derive(Debug)]
/// An OpenMode for listing RAR archive entries.
///
/// List mode will list all entries. The payload itself cannot be processed and instead can only
/// be skipped over. This will yield one header per individual file, regardless of how many parts
/// the file is split across.
pub struct List;
/// An OpenMode for listing RAR archive entries.
///
/// ListSplit mode will list all entries. The payload itself cannot be processed and instead can
/// only be skipped over. This will yield one header per individual file per part if the file is
/// split across multiple parts. The [`FileHeader::is_split`] method will return true in that case.
#[derive(Debug)]
pub struct ListSplit;

/// Mode with which the archive should be opened.
///
/// Possible modes are:
///
///    - [`List`](struct.List.html)
///    - [`ListSplit`](struct.ListSplit.html)
///    - [`Process`](struct.Process.html)
pub trait OpenMode: private::Sealed {
    const VALUE: private::OpenModeValue;
}
impl OpenMode for Process {
    const VALUE: private::OpenModeValue = private::OpenModeValue::Extract;
}
impl OpenMode for List {
    const VALUE: private::OpenModeValue = private::OpenModeValue::List;
}
impl OpenMode for ListSplit {
    const VALUE: private::OpenModeValue = private::OpenModeValue::ListIncSplit;
}

impl<Mode: OpenMode, C: Cursor> OpenArchive<Mode, C> {
    /// is the archive locked
    pub fn is_locked(&self) -> bool {
        self.flags.contains(ArchiveFlags::LOCK)
    }

    /// are the archive headers encrypted
    pub fn has_encrypted_headers(&self) -> bool {
        self.flags.contains(ArchiveFlags::ENC_HEADERS)
    }

    /// does the archive have a recovery record
    pub fn has_recovery_record(&self) -> bool {
        self.flags.contains(ArchiveFlags::RECOVERY)
    }

    /// does the archive have comments
    pub fn has_comment(&self) -> bool {
        self.flags.contains(ArchiveFlags::COMMENT)
    }

    /// is the archive solid (all files in a single compressed block).
    pub fn is_solid(&self) -> bool {
        self.flags.contains(ArchiveFlags::SOLID)
    }

    /// Volume information on the file that was *initially* opened.
    ///
    /// returns
    ///   - `VolumeInfo::None` if the opened file is a single-part archive
    ///   - `VolumeInfo::First` if the opened file is the first volume in a multipart archive
    ///   - `VolumeInfo::Subsequent` if the opened file is any other volume in a multipart archive
    ///
    /// Note that this value *never* changes from `First` to `Subsequent` by advancing to a
    /// different volume.
    pub fn volume_info(&self) -> VolumeInfo {
        if self.flags.contains(ArchiveFlags::FIRST_VOLUME) {
            VolumeInfo::First
        } else if self.flags.contains(ArchiveFlags::VOLUME) {
            VolumeInfo::Subsequent
        } else {
            VolumeInfo::None
        }
    }

    /// unsets the `damaged` flag so that `Iterator` will not refuse to yield elements.
    ///
    /// Normally, when an error is returned during iteration, the archive remembers this
    /// so that subsequent calls to `next` return `None` immediately. This is to prevent
    /// the same error from recurring over and over again, leading to endless loops in programs
    /// that might not have considered this. However, maybe there are errors that can be recovered
    /// from? That's where this method might come in handy if you really know what you're doing.
    /// However, should that be the case, I urge you to submit an issue / PR with an archive where
    /// the recoverable error can be reproduced so I can exclude that case from "irrecoverable
    /// errors" (currently all errors).
    ///
    /// Use at your own risk. Might be removed in future releases if somehow it can be verified
    /// which errors are recoverable and which are not.
    ///
    /// # Example how you *might* use this
    ///
    /// ```no_run
    /// use unrar_ng::{Archive, error::{When, Code}};
    ///
    /// let mut archive = Archive::new("corrupt.rar").open_for_listing().expect("archive error");
    /// loop {
    ///     let mut error = None;
    ///     for result in &mut archive {
    ///         match result {
    ///             Ok(entry) => println!("{entry}"),
    ///             Err(e) => error = Some(e),
    ///         }
    ///     }
    ///     match error {
    ///         // your special recoverable error, please submit a PR with reproducible archive
    ///         Some(e) if (e.when, e.code) == (When::Process, Code::BadData) => archive.force_heal(),
    ///         Some(e) => panic!("irrecoverable error: {e}"),
    ///         None => break,
    ///     }
    /// }
    /// ```
    pub fn force_heal(&mut self) {
        self.damaged = false;
    }
}

impl<Mode: OpenMode> OpenArchive<Mode, CursorBeforeHeader> {
    pub(crate) fn new(
        filename: &Path,
        password: Option<&[u8]>,
        recover: Option<&mut Option<Self>>,
    ) -> UnrarResult<Self> {
        let filename = pathed::construct(filename);

        // The guard is scoped to the FFI calls and nothing else. `Handle::drop` takes the same
        // lock, and both the `recover` path below and the failure arm can drop an
        // `OpenArchive` — under a held guard that would deadlock, so the archive is built
        // after the guard is gone.
        let (handle, open_result, flags) = {
            let _guard = serialised();

            let mut data =
                native::OpenArchiveDataEx::new(filename.as_ptr() as *const _, Mode::VALUE as u32);
            let handle =
                NonNull::new(unsafe { native::RAROpenArchiveEx(&mut data as *mut _) } as *mut _);

            // Part of the same transaction: the password belongs to the archive just opened.
            if let (Some(handle), Some(pw)) = (handle, password) {
                let cpw = std::ffi::CString::new(pw).unwrap();
                unsafe { native::RARSetPassword(handle.as_ptr(), cpw.as_ptr() as *const _) }
            }

            (handle, data.open_result, data.flags)
        };

        let arc = handle.map(|handle| OpenArchive {
            handle: Handle(handle),
            damaged: false,
            flags: ArchiveFlags::from_bits(flags).unwrap(),
            extra: CursorBeforeHeader,
            marker: std::marker::PhantomData,
        });
        let result = Code::from(open_result as i32);

        match (arc, result) {
            (Some(arc), Code::Success) => Ok(arc),
            (arc, _) => {
                recover.and_then(|recover| arc.and_then(|arc| recover.replace(arc)));
                Err(UnrarError::from(result, When::Open))
            }
        }
    }

    /// reads the next header of the underlying archive. The resulting OpenArchive will
    /// be in "ProcessFile" mode, i.e. the file corresponding to the header (that has just
    /// been read via this method call) will have to be read. Also contains header data
    /// via [`archive.entry()`](OpenArchive::entry).
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let archive = unrar_ng::Archive::new("data/version.rar").open_for_listing().unwrap().read_header();
    /// assert!(archive.as_ref().is_ok_and(Option::is_some));
    /// let archive = archive.unwrap().unwrap();
    /// assert_eq!(archive.entry().filename.as_os_str(), "VERSION");
    /// ```
    pub fn read_header(self) -> UnrarResult<Option<OpenArchive<Mode, CursorBeforeFile>>> {
        Ok(read_header(&self.handle)?.map(|entry| OpenArchive {
            extra: CursorBeforeFile { header: entry },
            damaged: self.damaged,
            handle: self.handle,
            flags: self.flags,
            marker: std::marker::PhantomData,
        }))
    }
}

impl OpenArchive<Process, CursorBeforeHeader> {
    /// Extracts all files from the archive to the specified directory in a single operation.
    ///
    /// This method is significantly faster than iterating through files individually,
    /// especially for archives containing many small files. It bypasses the per-file
    /// FFI overhead by using a batch extraction function internally.
    ///
    /// # Arguments
    ///
    /// * `dest` - The destination directory path. If the directory doesn't exist,
    ///   it will be created. Pass an empty path or `"."` for current directory.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use unrar_ng::Archive;
    ///
    /// let archive = Archive::new("archive.rar")
    ///     .open_for_processing()
    ///     .expect("Failed to open archive");
    ///
    /// archive.extract_all("./output")
    ///     .expect("Failed to extract archive");
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if `dest` contains nul characters.
    pub fn extract_all<P: AsRef<Path>>(self, dest: P) -> UnrarResult<()> {
        crate::locale::ensure_initialized();
        let dest_path = pathed::construct(dest.as_ref());
        let result = pathed::extract_all(self.handle.0.as_ptr(), &dest_path);
        match Code::from(result) {
            Code::Success => Ok(()),
            code => Err(UnrarError::from(code, When::Process)),
        }
    }

    /// Extracts all files from the archive with a progress callback.
    ///
    /// This method is similar to [`extract_all`](Self::extract_all) but allows you to
    /// receive notifications about extraction progress through a callback function.
    ///
    /// # Arguments
    ///
    /// * `dest` - The destination directory path.
    /// * `callback` - A closure that receives [`ExtractEvent`] notifications.
    ///   For `Start`, `Ok` and `Err`, returning `false` cancels the rest
    ///   of the extraction and the call returns `Ok(ExtractStatus::Cancelled)`.
    ///   For `LargeDictWarning`, `false` rejects the oversized dictionary
    ///   instead of cancelling — extraction then fails with
    ///   [`Code::LargeDict`](crate::error::Code::LargeDict).
    ///
    /// # Returns
    ///
    /// * `Ok(ExtractStatus::Completed)` — the DLL finished iterating the
    ///   archive without the callback ever returning `false` from a
    ///   cancellable event.
    /// * `Ok(ExtractStatus::Cancelled)` — the callback aborted extraction
    ///   early on a `Start`/`Ok`/`Err` event.
    /// * `Err(UnrarError { .. })` — the DLL surfaced an error (incl.
    ///   [`Code::LargeDict`](crate::error::Code::LargeDict) when the
    ///   callback rejected an oversized dictionary).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use unrar_ng::{Archive, ExtractEvent, ExtractStatus};
    ///
    /// let archive = Archive::new("archive.rar")
    ///     .open_for_processing()
    ///     .expect("Failed to open archive");
    ///
    /// let status = archive.extract_all_with_callback("./output", |event| {
    ///     match event {
    ///         ExtractEvent::Start { filename, size } => {
    ///             print!("extracting {}... ({} bytes) ", filename.display(), size);
    ///             true // continue extraction
    ///         }
    ///         ExtractEvent::Ok { filename, size } => {
    ///             println!("ok ({} bytes) {}", size, filename.display());
    ///             true
    ///         }
    ///         ExtractEvent::Err { filename, error_code } => {
    ///             println!("error (code: {})", error_code);
    ///             true // continue with other files
    ///         }
    ///         ExtractEvent::LargeDictWarning { dict_size_kb, max_dict_size_kb } => {
    ///             eprintln!("archive needs {dict_size_kb} KB dict, build supports {max_dict_size_kb} KB");
    ///             false // refuse oversized dict; extraction fails with Code::LargeDict
    ///         }
    ///         _ => true,
    ///     }
    /// }).expect("Failed to extract archive");
    /// match status {
    ///     ExtractStatus::Completed => println!("done"),
    ///     ExtractStatus::Cancelled => println!("user cancelled"),
    ///     _ => {}
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if `dest` contains nul characters.
    pub fn extract_all_with_callback<P, F>(
        self,
        dest: P,
        mut callback: F,
    ) -> UnrarResult<ExtractStatus>
    where
        P: AsRef<Path>,
        F: FnMut(ExtractEvent) -> bool,
    {
        crate::locale::ensure_initialized();

        // Userdata struct to pass to the C callback
        struct CallbackData<'a, F> {
            callback: &'a mut F,
            cancelled: bool,
            // Carried from UCM_EXTRACTFILE (Start) into UCM_EXTRACTFILE_OK
            // (Ok) so the surfaced ExtractEvent::Ok can include the size
            // the upstream library only reports at Start time.
            pending_size: u64,
        }

        extern "C" fn extract_callback<F>(
            msg: native::UINT,
            user_data: native::LPARAM,
            p1: native::LPARAM,
            p2: native::LPARAM,
        ) -> c_int
        where
            F: FnMut(ExtractEvent) -> bool,
        {
            if user_data == 0 {
                return 0;
            }
            let data = unsafe { &mut *(user_data as *mut CallbackData<'_, F>) };

            // Helper to read a wchar_t* string from p1.
            // native::WCHAR is i32 on Unix, u16 on Windows.
            //
            // FIXME: the `char::from_u32 + filter_map` decode below silently
            // drops unpaired surrogates on Windows (wchar_t = u16), so the
            // filename reported via ExtractEvent can diverge from what
            // pathed/all.rs writes to disk via lossless WideCString. The
            // 2048-wchar truncation cap below is also a known gap — both
            // problems disappear once this is rewritten with
            // U16/U32CString::from_ptr_truncate -> OsString -> PathBuf.
            fn read_filename(p1: native::LPARAM) -> Option<PathBuf> {
                if p1 == 0 {
                    return None;
                }

                let ptr = p1 as *const native::WCHAR;
                if ptr.is_null() {
                    return None;
                }

                // Find null terminator. 2048 mirrors the upstream maximum
                // path length used by `WideCString::from_ptr_truncate` in
                // `pathed/all.rs::construct` and elsewhere — see the comment
                // on `Internal::callback`'s UCM_CHANGEVOLUMEW arm in this file.
                let mut len = 0usize;
                const MAX_LEN: usize = 2048;
                unsafe {
                    while len < MAX_LEN && *ptr.add(len) != 0 {
                        len += 1;
                    }
                }

                if len == 0 {
                    return None;
                }

                // Convert wchar_t slice to PathBuf
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

                // wchar_t is i32 on Unix, u16 on Windows
                // Convert to chars respecting the platform's wchar_t representation
                let path_string: String = slice
                    .iter()
                    .filter_map(|&c| char::from_u32(c as u32))
                    .collect();

                Some(PathBuf::from(path_string))
            }

            match msg {
                native::UCM_EXTRACTFILE => {
                    // p1 = filename (wchar_t*), p2 = file size
                    let size = p2 as u64;
                    // Stash the size unconditionally so a later OK event
                    // still gets it even if read_filename fails here and
                    // we end up skipping the Start callback.
                    data.pending_size = size;
                    if let Some(filename) = read_filename(p1) {
                        let event = ExtractEvent::Start { filename, size };
                        if !(data.callback)(event) {
                            data.cancelled = true;
                            return -1; // Cancel extraction
                        }
                    }
                    0
                }
                native::UCM_EXTRACTFILE_OK => {
                    // p1 = filename (wchar_t*), p2 = 0
                    if let Some(filename) = read_filename(p1) {
                        let event = ExtractEvent::Ok {
                            filename,
                            size: data.pending_size,
                        };
                        if !(data.callback)(event) {
                            data.cancelled = true;
                            return -1;
                        }
                    }
                    0
                }
                native::UCM_EXTRACTFILE_ERR => {
                    // p1 = filename (wchar_t*), p2 = error code
                    if let Some(filename) = read_filename(p1) {
                        let event = ExtractEvent::Err {
                            filename,
                            error_code: p2 as i32,
                        };
                        if !(data.callback)(event) {
                            data.cancelled = true;
                            return -1;
                        }
                    }
                    0
                }
                native::UCM_LARGEDICT => {
                    // Upstream `uiDictLimit` (vendor/unrar/uisilent.cpp): only
                    // a return of 1 lets extraction continue; any other value
                    // (including 0) makes the DLL fail with ERAR_LARGE_DICT.
                    // P1/P2 are the required and max dict sizes in KB.
                    //
                    // Rejecting an oversized dictionary is not a user cancel —
                    // it surfaces as `Err(Code::LargeDict)`, so we deliberately
                    // do NOT set `data.cancelled` here. That keeps
                    // `ExtractStatus::Cancelled` strictly meaning "callback
                    // returned false from Start/Ok/Err".
                    let event = ExtractEvent::LargeDictWarning {
                        dict_size_kb: p1 as u64,
                        max_dict_size_kb: p2 as u64,
                    };
                    if (data.callback)(event) { 1 } else { 0 }
                }
                native::UCM_CHANGEVOLUMEW => {
                    // Handle volume change: -1 means stop (volume not found)
                    match p2 {
                        native::RAR_VOL_ASK => -1,
                        _ => 0,
                    }
                }
                _ => 0,
            }
        }

        let dest_path = pathed::construct(dest.as_ref());

        let mut callback_data = CallbackData {
            callback: &mut callback,
            cancelled: false,
            pending_size: 0,
        };

        // One transaction, and the user's callback runs under the lock — see [`DLL`]: it must
        // not re-enter this crate.
        let _guard = serialised();

        unsafe {
            native::RARSetCallback(
                self.handle.0.as_ptr(),
                Some(extract_callback::<F>),
                &mut callback_data as *mut _ as native::LPARAM,
            );
        }

        let result = pathed::extract_all(self.handle.0.as_ptr(), &dest_path);

        match Code::from(result) {
            Code::Success if callback_data.cancelled => Ok(ExtractStatus::Cancelled),
            Code::Success => Ok(ExtractStatus::Completed),
            code => Err(UnrarError::from(code, When::Process)),
        }
    }
}

impl Iterator for OpenArchive<List, CursorBeforeHeader> {
    type Item = Result<FileHeader, UnrarError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.damaged {
            return None;
        }
        match read_header(&self.handle) {
            Ok(Some(header)) => {
                match Internal::<Skip>::process_file_raw(&self.handle, None, None) {
                    Ok(_) => Some(Ok(header)),
                    Err(s) => {
                        self.damaged = true;
                        Some(Err(s))
                    }
                }
            }
            Ok(None) => None,
            Err(s) => {
                self.damaged = true;
                Some(Err(s))
            }
        }
    }
}

impl Iterator for OpenArchive<ListSplit, CursorBeforeHeader> {
    type Item = Result<FileHeader, UnrarError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.damaged {
            return None;
        }
        match read_header(&self.handle) {
            Ok(Some(header)) => {
                match Internal::<Skip>::process_file_raw(&self.handle, None, None) {
                    Ok(_) => Some(Ok(header)),
                    Err(s) => {
                        self.damaged = true;
                        Some(Err(s))
                    }
                }
            }
            Ok(None) => None,
            Err(s) => {
                self.damaged = true;
                Some(Err(s))
            }
        }
    }
}

impl<M: OpenMode> OpenArchive<M, CursorBeforeFile> {
    /// returns the file header for the file that follows which is to be processed next.
    pub fn entry(&self) -> &FileHeader {
        &self.extra.header
    }

    /// skips over the next file, not doing anything with it.
    pub fn skip(self) -> UnrarResult<OpenArchive<M, CursorBeforeHeader>> {
        self.process_file::<Skip>(None, None)
    }

    fn process_file<PM: ProcessMode>(
        self,
        path: Option<&pathed::RarStr>,
        file: Option<&pathed::RarStr>,
    ) -> UnrarResult<OpenArchive<M, CursorBeforeHeader>>
    where
        PM::Output: Default,
    {
        Ok(self.process_file_x::<PM>(path, file)?.1)
    }

    fn process_file_x<PM: ProcessMode>(
        self,
        path: Option<&pathed::RarStr>,
        file: Option<&pathed::RarStr>,
    ) -> UnrarResult<(PM::Output, OpenArchive<M, CursorBeforeHeader>)>
    where
        PM::Output: Default,
    {
        let result = Ok((
            Internal::<PM>::process_file_raw(&self.handle, path, file)?,
            OpenArchive {
                extra: CursorBeforeHeader,
                damaged: self.damaged,
                handle: self.handle,
                flags: self.flags,
                marker: std::marker::PhantomData,
            },
        ));
        result
    }
}

impl OpenArchive<Process, CursorBeforeFile> {
    /// Reads the underlying file into a `Vec<u8>`
    /// Returns the data as well as the owned Archive that can be processed further.
    pub fn read(self) -> UnrarResult<(Vec<u8>, OpenArchive<Process, CursorBeforeHeader>)> {
        let (bytes, result) = self.read_into(Vec::new());
        result.map(|archive| (bytes, archive))
    }

    /// Reads the underlying file, handing each chunk to `sink` as the DLL produces it.
    ///
    /// Nothing is written to disk: this runs the DLL's test operation and captures the
    /// bytes through the data callback.
    ///
    /// The sink comes back either way, so a sink that stopped early still carries what it
    /// took. The archive comes back only on success — [`DataSink::write_chunk`] returning
    /// `false` unwinds the DLL's extraction, which leaves the handle's cursor somewhere
    /// this wrapper cannot describe, so continuing from it is not offered. An aborted read
    /// is [`Code::Aborted`](crate::error::Code::Aborted) rather than an error the DLL
    /// raised.
    pub fn read_into<S: DataSink + core::fmt::Debug>(
        self,
        sink: S,
    ) -> (S, UnrarResult<OpenArchive<Process, CursorBeforeHeader>>) {
        let (sink, result) =
            Internal::<ReadToSink<S>>::process_file_raw_with(&self.handle, None, None, sink);
        let archive = OpenArchive {
            extra: CursorBeforeHeader,
            damaged: self.damaged,
            handle: self.handle,
            flags: self.flags,
            marker: std::marker::PhantomData,
        };
        (sink, result.map(|()| archive))
    }

    /// Test the file without extracting it
    pub fn test(self) -> UnrarResult<OpenArchive<Process, CursorBeforeHeader>> {
        Ok(self.process_file::<Test>(None, None)?)
    }

    /// Extracts the file into the current working directory
    /// Returns the OpenArchive for further processing
    pub fn extract(self) -> UnrarResult<OpenArchive<Process, CursorBeforeHeader>> {
        self.dir_extract(None)
    }

    /// Extracts the file into the specified directory.  
    /// Returns the OpenArchive for further processing
    ///
    /// # Panics
    ///
    /// This function will panic if `base` contains nul characters.
    pub fn extract_with_base<P: AsRef<Path>>(
        self,
        base: P,
    ) -> UnrarResult<OpenArchive<Process, CursorBeforeHeader>> {
        self.dir_extract(Some(base.as_ref()))
    }

    /// Extracts the file into the specified file.
    /// Returns the OpenArchive for further processing
    ///
    /// # Panics
    ///
    /// This function will panic if `dest` contains nul characters.
    pub fn extract_to<P: AsRef<Path>>(
        self,
        file: P,
    ) -> UnrarResult<OpenArchive<Process, CursorBeforeHeader>> {
        let dest = pathed::construct(file.as_ref());
        self.process_file::<Extract>(None, Some(&dest))
    }

    /// extracting into a directory if the filename has unicode characters
    /// does not work on Linux, so we must specify the full path for Linux
    fn dir_extract(
        self,
        base: Option<&Path>,
    ) -> UnrarResult<OpenArchive<Process, CursorBeforeHeader>> {
        let (path, file) = pathed::preprocess_extract(base, &self.entry().filename);
        self.process_file::<Extract>(path.as_deref(), file.as_deref())
    }
}

/// The longest entry name libunrar will produce, plus its terminator.
///
/// `MAXPATHSIZE` (`vendor/unrar/rardefs.hpp`) is the ceiling `arcread.cpp` clamps a stored
/// name to before it is ever decoded, so a buffer of that size cannot be overrun by a name
/// this library can return. Sized to the ceiling rather than to something merely generous,
/// because the alternative is a silent cut — see [`read_header`].
const MAX_NAME_WCHARS: usize = 0x10000 + 1;

fn read_header(handle: &Handle) -> UnrarResult<Option<FileHeader>> {
    // One transaction: the callback registration and the read that consumes it. A second
    // thread registering its own between the two would redirect this one's.
    let _guard = serialised();

    let mut userdata: Userdata<<Skip as ProcessMode>::Output> = Default::default();
    unsafe {
        native::RARSetCallback(
            handle.0.as_ptr(),
            Some(Internal::<Skip>::callback),
            &mut userdata as *mut _ as native::LPARAM,
        );
    }

    // `HeaderDataEx::FileNameW` is a fixed `[wchar_t; 1024]`, and `dll.cpp` fills it with
    // `wcsncpyz(.., ASIZE(D->FileNameW))` — so a stored name longer than 1023 characters came
    // back cut, with nothing to say it had been. A caller that keys on the extension then saw
    // a name with no extension and passed the entry over: a file silently missing from the
    // archive it is listing.
    //
    // `FileNameEx` is the escape hatch the DLL already provides. `dll.cpp` fills it whenever
    // it is non-null, bounded by `FileNameExSize` rather than by a fixed array, and this
    // buffer is sized to libunrar's own `MAXPATHSIZE` ceiling so the bound cannot bite.
    let mut long_name = vec![0 as native::WcharT; MAX_NAME_WCHARS];
    let mut header = native::HeaderDataEx {
        file_name_ex: long_name.as_mut_ptr(),
        file_name_ex_size: MAX_NAME_WCHARS as c_uint,
        ..Default::default()
    };

    let read_result =
        Code::from(unsafe { native::RARReadHeaderEx(handle.0.as_ptr(), &mut header as *mut _) });
    match read_result {
        Code::Success => {
            let mut entry = FileHeader::from(header);
            // Empty only if the DLL declined to fill it, in which case the short field — cut
            // or not — is still better than no name at all.
            let full = unsafe {
                widestring::WideCString::from_ptr_truncate(
                    long_name.as_ptr() as *const _,
                    MAX_NAME_WCHARS,
                )
            };
            if !full.is_empty() {
                entry.filename = PathBuf::from(full.to_os_string());
            }
            Ok(Some(entry))
        }
        Code::EndArchive => Ok(None),
        _ => Err(UnrarError::from(read_result, When::Read)),
    }
}

/// Where an entry's bytes go as libunrar produces them.
///
/// The DLL pushes an entry through `UCM_PROCESSDATA` in chunks rather than handing over a
/// reader the caller drives, so this is the only seam at which a caller can see the bytes
/// — and the only one at which it can decline the rest of them.
///
/// Returning `false` aborts the read. That matters for a caller that has to bound what it
/// holds: without it, the only way to stop an entry that turns out to be larger than the
/// caller can accept is to let the whole thing arrive first, which is not a bound.
/// libunrar has supported this the whole time — `vendor/unrar/rdwrfn.cpp` calls
/// `ErrHandler.Exit(RARX_USERBREAK)` when the callback returns `-1` — and only this
/// wrapper's hardcoded `0` stood in the way.
pub trait DataSink {
    /// Takes one chunk. Returns `false` to abort the read.
    fn write_chunk(&mut self, chunk: &[u8]) -> bool;
}

/// Accumulates the whole entry, which is what [`OpenArchive::read`] does.
impl DataSink for Vec<u8> {
    fn write_chunk(&mut self, chunk: &[u8]) -> bool {
        self.extend_from_slice(chunk);
        true
    }
}

#[derive(Debug)]
struct Skip;
#[derive(Debug)]
struct ReadToSink<S>(std::marker::PhantomData<S>);
#[derive(Debug)]
struct Extract;
#[derive(Debug)]
struct Test;

trait ProcessMode: core::fmt::Debug {
    const OPERATION: private::Operation;
    type Output: core::fmt::Debug;

    /// Returns `false` to abort the operation.
    fn process_data(data: &mut Self::Output, other: &[u8]) -> bool;
}
impl ProcessMode for Skip {
    const OPERATION: private::Operation = private::Operation::Skip;
    type Output = ();

    fn process_data(_: &mut Self::Output, _: &[u8]) -> bool {
        true
    }
}
impl<S: DataSink + core::fmt::Debug> ProcessMode for ReadToSink<S> {
    // `Test`, not `Extract`: the bytes are captured through the callback and nothing is
    // written to disk.
    const OPERATION: private::Operation = private::Operation::Test;
    type Output = S;

    fn process_data(sink: &mut Self::Output, other: &[u8]) -> bool {
        sink.write_chunk(other)
    }
}
impl ProcessMode for Extract {
    const OPERATION: private::Operation = private::Operation::Extract;
    type Output = ();

    fn process_data(_: &mut Self::Output, _: &[u8]) -> bool {
        true
    }
}
impl ProcessMode for Test {
    const OPERATION: private::Operation = private::Operation::Test;
    type Output = ();

    fn process_data(_: &mut Self::Output, _: &[u8]) -> bool {
        true
    }
}

struct Internal<M: ProcessMode> {
    marker: std::marker::PhantomData<M>,
}

impl<M: ProcessMode> Internal<M> {
    extern "C" fn callback(
        msg: native::UINT,
        user_data: native::LPARAM,
        p1: native::LPARAM,
        p2: native::LPARAM,
    ) -> c_int {
        if user_data == 0 {
            return 0;
        }
        let user_data = unsafe { &mut *(user_data as *mut Userdata<M::Output>) };
        match msg {
            native::UCM_CHANGEVOLUMEW => {
                // 2048 seems to be the buffer size in unrar,
                // also it's the maximum path length since 5.00.
                let next =
                    unsafe { widestring::WideCString::from_ptr_truncate(p1 as *const _, 2048) };
                user_data.1 = Some(next);
                match p2 {
                    // Next volume not found. -1 means stop
                    native::RAR_VOL_ASK => -1,
                    // Next volume found, 0 means continue
                    _ => 0,
                }
            }
            native::UCM_PROCESSDATA => {
                let raw_slice = std::ptr::slice_from_raw_parts(p1 as *const u8, p2 as _);
                if M::process_data(&mut user_data.0, unsafe { &*raw_slice as &_ }) {
                    0
                } else {
                    // Recorded because the DLL will not tell us: `RARX_USERBREAK` has no
                    // `RarErrorToDll` case and comes back as `ERAR_UNKNOWN`.
                    user_data.2 = true;
                    -1
                }
            }
            _ => 0,
        }
    }

    fn process_file_raw(
        handle: &Handle,
        path: Option<&pathed::RarStr>,
        file: Option<&pathed::RarStr>,
    ) -> UnrarResult<M::Output>
    where
        M::Output: Default,
    {
        let (output, result) =
            Self::process_file_raw_with(handle, path, file, M::Output::default());
        result.map(|()| output)
    }

    /// As [`Self::process_file_raw`], but the caller supplies the output and gets it back
    /// whichever way the call went.
    ///
    /// Returning it on the error path is the point: a sink that aborted holds what it read
    /// before it stopped, and a sink the caller configured is not reconstructible from
    /// `Default`.
    fn process_file_raw_with(
        handle: &Handle,
        path: Option<&pathed::RarStr>,
        file: Option<&pathed::RarStr>,
        output: M::Output,
    ) -> (M::Output, UnrarResult<()>) {
        // One transaction: registering the callback and running the operation that uses it.
        // The sink runs under this lock — see [`DLL`]: it must not re-enter this crate.
        let _guard = serialised();

        let mut user_data: Userdata<M::Output> = (output, None, false);
        unsafe {
            native::RARSetCallback(
                handle.0.as_ptr(),
                Some(Self::callback),
                &mut user_data as *mut _ as native::LPARAM,
            );
        }
        let process_result = Code::from(pathed::process_file(
            handle.0.as_ptr(),
            M::OPERATION as i32,
            path,
            file,
        ));
        let result = match process_result {
            Code::Success => Ok(()),
            // Our own abort, recovered before the DLL's `ERAR_UNKNOWN` can be mistaken for
            // a real failure.
            _ if user_data.2 => Err(UnrarError::from(Code::Aborted, When::Process)),
            _ => Err(UnrarError::from(process_result, When::Process)),
        };
        (user_data.0, result)
    }
}

bitflags::bitflags! {
    #[derive(Debug)]
    struct EntryFlags: u32 {
        const SPLIT_BEFORE = 0x1;
        const SPLIT_AFTER = 0x2;
        const ENCRYPTED = 0x4;
        // const RESERVED = 0x8;
        const SOLID = 0x10;
        const DIRECTORY = 0x20;
    }
}

/// Metadata for an entry in a RAR archive
///
/// Created using the read_header methods in an OpenArchive, contains
/// information for the file that follows which is to be processed next.
#[allow(missing_docs)]
#[derive(Debug)]
pub struct FileHeader {
    pub filename: PathBuf,
    flags: EntryFlags,
    pub unpacked_size: u64,
    pub file_crc: u32,
    pub file_time: u32,
    pub method: u32,
    pub file_attr: u32,
}

impl FileHeader {
    /// is this entry split across multiple volumes.
    ///
    /// Will also work in open mode [`List`]
    pub fn is_split(&self) -> bool {
        self.flags.contains(EntryFlags::SPLIT_BEFORE)
            || self.flags.contains(EntryFlags::SPLIT_AFTER)
    }

    /// is this entry split across multiple volumes, starting here
    ///
    /// Will also work in open mode [`List`]
    pub fn is_split_after(&self) -> bool {
        self.flags.contains(EntryFlags::SPLIT_AFTER)
    }

    /// is this entry split across multiple volumes, starting here
    ///
    /// Will always return false in open mode [`List`][^1].
    ///
    /// [^1]: this claim is not proven, however, the DLL seems to always skip
    /// files where this flag would have been set.
    pub fn is_split_before(&self) -> bool {
        self.flags.contains(EntryFlags::SPLIT_BEFORE)
    }

    /// is this entry a directory
    pub fn is_directory(&self) -> bool {
        self.flags.contains(EntryFlags::DIRECTORY)
    }

    /// is this entry encrypted
    pub fn is_encrypted(&self) -> bool {
        self.flags.contains(EntryFlags::ENCRYPTED)
    }

    /// is this entry a file
    pub fn is_file(&self) -> bool {
        !self.is_directory()
    }
}

impl fmt::Display for FileHeader {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.filename)?;
        if self.is_directory() {
            write!(f, "/")?
        }
        if self.is_split() {
            write!(f, " (partial)")?
        }
        Ok(())
    }
}

impl From<native::HeaderDataEx> for FileHeader {
    fn from(header: native::HeaderDataEx) -> Self {
        // `native::HeaderDataEx` is `#[repr(C, packed(1))]` to match the C++
        // `#pragma pack(push, 1)` layout, so taking `&header.filename_w` is
        // forbidden. `&raw const` produces a raw pointer without creating a
        // reference; the underlying wchar_t array happens to be 4-byte
        // aligned at its real offset in memory, so the subsequent read is
        // fine for `WideCString::from_ptr_truncate`.
        let filename_w_ptr = &raw const header.filename_w as *const _;
        let filename =
            unsafe { widestring::WideCString::from_ptr_truncate(filename_w_ptr, 1024) };

        // Packed-struct fields are `Copy` primitives here, so value reads
        // are legal; we copy each scalar out into a local before passing it
        // to the constructor so rustc can't be tempted into taking a
        // reference to the packed field.
        let flags = header.flags;
        let unp_size = header.unp_size;
        let unp_size_high = header.unp_size_high;
        let file_crc = header.file_crc;
        let file_time = header.file_time;
        let method = header.method;
        let file_attr = header.file_attr;

        FileHeader {
            filename: PathBuf::from(filename.to_os_string()),
            flags: EntryFlags::from_bits(flags).unwrap(),
            unpacked_size: unpack_unp_size(unp_size, unp_size_high),
            file_crc,
            file_time,
            method,
            file_attr,
        }
    }
}

fn unpack_unp_size(unp_size: c_uint, unp_size_high: c_uint) -> u64 {
    ((unp_size_high as u64) << (8 * std::mem::size_of::<c_uint>())) | (unp_size as u64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn combine_size() {
        use super::unpack_unp_size;
        let (high, low) = (1u32, 1464303715u32);
        assert_eq!(unpack_unp_size(low, high), 5759271011);
    }
}
