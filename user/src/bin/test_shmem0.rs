#![no_std]
#![no_main]

#[macro_use]
extern crate user_lib;

use user_lib::{mmap, munmap};

/*
理想结果：输出 Test 06_1 OK!
*/

#[no_mangle]
pub fn main() -> i32 {
    let start0: usize = 0x10000000;
    let start1: usize = 0x10001000;
    let len: usize = 4096;
    let prot: usize = 3;
    let shmem_id = mmap(start0, len, prot, 1, -1);
    let addr0: *mut u8 = start0 as *mut u8;
    let shmem_id = mmap(start1, len, prot, 1, shmem_id);
    let addr1: *mut u8 = start1 as *mut u8;
    unsafe {
        *addr0 = 97;
    }
    assert!(unsafe { *addr1 == 97 });
    munmap(start0, len);
    munmap(start1, len);
    0
}
