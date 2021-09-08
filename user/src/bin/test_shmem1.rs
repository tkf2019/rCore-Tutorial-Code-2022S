#![no_std]
#![no_main]

#[macro_use]
extern crate user_lib;

use user_lib::{exit, fork, mmap, munmap, wait};

#[no_mangle]
pub fn main() -> i32 {
    let start: usize = 0x10000000;
    let len: usize = 4096;
    let prot: usize = 3;
    let shmem_id = mmap(start, len, prot, 1, -1);

    let pid = fork();
    if pid != 0 {
        // parent
        let addr: *mut u8 = start as *mut u8;
        unsafe {
            *addr = 97;
        }
        let mut exit_code: i32 = 0;
        wait(&mut exit_code);
        munmap(start, len);
    } else {
        // child
        let start: usize = 0x10001000;
        mmap(start, len, prot, 1, shmem_id);
        let addr: *mut u8 = start as *mut u8;
        assert!(unsafe { *addr == 97 });
        munmap(start, len);
        exit(0);
    }
    println!("Test 06_2 OK!");
    0
}
