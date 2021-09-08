#![no_std]
#![no_main]

#[macro_use]
extern crate user_lib;

use user_lib::{exit, fork, getpid, mmap, munmap, wait};

fn check_child(start: usize, len: usize, prot: usize, shmem_id: isize, offset: usize, pid: isize) {
    let p = getpid();
    mmap(start + offset, len, prot, 1, shmem_id);
    let mut exit_code: i32 = 0;
    wait(&mut exit_code);
    let addr: *mut u8 = (start + offset) as *mut u8;
    unsafe {
        assert!(*addr == (pid as u8));
        *addr = p as u8;
    }
    munmap(start, len);
    exit(0);
}

#[no_mangle]
pub fn main() -> i32 {
    let start: usize = 0x10000000;
    let len: usize = 4096;
    let prot: usize = 3;
    let shmem_id = mmap(start, len, prot, 1, -1);

    let pid = fork();
    if pid != 0 {
        // root
        let mut exit_code: i32 = 0;
        wait(&mut exit_code);
        let addr: *mut u8 = start as *mut u8;
        unsafe {
            assert!(*addr == (pid as u8));
        }
        munmap(start, len);
    } else {
        let pid = fork();
        if pid != 0 {
            // child 1
            check_child(start, len, prot, shmem_id, 0x1000, pid);
        } else {
            let pid = fork();
            if pid != 0 {
                // child 2
                check_child(start, len, prot, shmem_id, 0x2000, pid);
            } else {
                let pid = fork();
                if pid != 0 {
                    // child 3
                    check_child(start, len, prot, shmem_id, 0x3000, pid);
                } else {
                    // child 4
                    let p = getpid() as u8;
                    mmap(start + 0x4000, len, prot, 1, shmem_id);
                    let addr: *mut u8 = (start + 0x4000) as *mut u8;
                    unsafe {
                        *addr = p;
                    }
                    munmap(start, len);
                    exit(0);
                }
            }
        }
    }

    println!("Test 06_3 OK!");
    0
}
