trait T {} struct A; impl T for A {} struct B; impl T for B {} fn main() { let v: Vec<Box<dyn T>> = vec![Box::new(A), Box::new(B)]; }
