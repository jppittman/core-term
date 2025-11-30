use std::sync::{Arc, OnceLock};
use std::marker::PhantomData;
use pixelflow_core::pipe::Surface;
use pixelflow_core::batch::Batch;

pub trait Factory<D>: Send + Sync {
    fn call(&self) -> D;
}

impl<F, D> Factory<D> for F
where
    F: Fn() -> D + Send + Sync,
{
    fn call(&self) -> D {
        (self)()
    }
}

pub trait Mapper<D, S>: Send + Sync {
    fn map(&self, data: &D) -> S;
}

impl<M, D, S> Mapper<D, S> for M
where
    M: Fn(&D) -> S + Send + Sync,
{
    fn map(&self, data: &D) -> S {
        (self)(data)
    }
}

#[derive(Clone)]
pub struct Lazy<F, M, D, S> {
    factory: F,
    mapper: M,
    cache: Arc<OnceLock<D>>,
    _phantom: PhantomData<S>,
}

impl<F, M, D, S> Lazy<F, M, D, S> {
    pub fn new(factory: F, mapper: M) -> Self {
        Self {
            factory,
            mapper,
            cache: Arc::new(OnceLock::new()),
            _phantom: PhantomData,
        }
    }
}

impl<F, M, D, S, P> Surface<P> for Lazy<F, M, D, S>
where
    F: Factory<D>,
    M: Mapper<D, S>,
    D: Send + Sync,
    S: Surface<P>,
    P: Copy,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let data = self.cache.get_or_init(|| self.factory.call());
        let surface = self.mapper.map(data);
        surface.eval(x, y)
    }
}
