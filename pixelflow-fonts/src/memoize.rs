use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::hash::Hash;

pub trait MemoizeFactory<K, V>: Send + Sync {
    fn call(&self, key: K) -> V;
}

impl<F, K, V> MemoizeFactory<K, V> for F where F: Fn(K) -> V + Send + Sync {
    fn call(&self, key: K) -> V { (self)(key) }
}

impl<K, V> MemoizeFactory<K, V> for Arc<dyn Fn(K) -> V + Send + Sync> {
    fn call(&self, key: K) -> V { (**self)(key) }
}

#[derive(Clone)]
pub struct Memoize<K, V, F> {
    cache: Arc<RwLock<HashMap<K, V>>>,
    factory: F,
}

impl<K, V, F> Memoize<K, V, F>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
    F: MemoizeFactory<K, V>,
{
    pub fn new(factory: F) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            factory,
        }
    }

    pub fn get(&self, key: K) -> V {
        if let Some(v) = self.cache.read().unwrap().get(&key) {
            return v.clone();
        }
        let mut map = self.cache.write().unwrap();
        if let Some(v) = map.get(&key) {
            return v.clone();
        }
        let v = self.factory.call(key.clone());
        map.insert(key, v.clone());
        v
    }
}
