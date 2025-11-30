use std::sync::{Arc, OnceLock};
use pixelflow_core::pipe::Surface;
use pixelflow_core::batch::Batch;

pub trait Factory<S>: Send + Sync {
    fn call(&self) -> S;
}

impl<F, S> Factory<S> for F
where
    F: Fn() -> S + Send + Sync,
{
    fn call(&self) -> S {
        (self)()
    }
}

// Implement for Arc<dyn Fn> to ensure it works even if Arc doesn't impl Fn directly
impl<'a, S> Factory<S> for Arc<dyn Fn() -> S + Send + Sync + 'a> {
    fn call(&self) -> S {
        (self)()
    }
}

#[derive(Clone)]
pub struct Lazy<F, S> {
    factory: F,
    cache: Arc<OnceLock<S>>,
}

impl<F, S> Lazy<F, S> {
    pub fn new(factory: F) -> Self {
        Self {
            factory,
            cache: Arc::new(OnceLock::new()),
        }
    }
}

impl<F, S, P> Surface<P> for Lazy<F, S>
where
    F: Factory<S>,
    S: Surface<P> + Send + Sync,
    P: Copy,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let surface = self.cache.get_or_init(|| self.factory.call());
        surface.eval(x, y)
    }
}
