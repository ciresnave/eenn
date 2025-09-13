use std::{collections::HashMap, sync::Arc};

// Type alias to reduce clippy type-complexity warnings for factory functions.
type Factory = Box<dyn Fn(&[f32]) -> Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static> + Send + Sync>;

// Store either “known ops” (stateless fn pointers) or closures with metadata.
pub struct FuncMeta {
    f: Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>,
    doc: &'static str,
}

impl FuncMeta {
    /// Create a new FuncMeta from any Fn (function pointer or closure).
    pub fn new<F>(f: F, doc: &'static str) -> Self
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        Self {
            f: Arc::new(f),
            doc,
        }
    }

    /// Return a clone of the stored function as an Arc.
    pub fn function(&self) -> Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static> {
        Arc::clone(&self.f)
    }

    pub fn doc(&self) -> &'static str {
        self.doc
    }
}

/// Example usage:
///
/// ```
/// use eenn::{FunctionRegistry, relu, scale};
/// let mut r = FunctionRegistry::empty();
/// r.register_fn("relu", relu, "ReLU");
/// r.register("scale_075", scale(0.75), "Scale by 0.75");
/// let f = r.get("scale_075").expect("found");
/// assert_eq!((f)(4.0f32), 3.0f32);
/// ```
pub struct FunctionRegistry {
    registry: HashMap<&'static str, FuncMeta>,
    // factories: name -> factory(params) -> Arc<dyn Fn(f32) -> f32>
    factories: HashMap<&'static str, Factory>,
}

impl FunctionRegistry {
    pub fn new(registry: HashMap<&'static str, FuncMeta>) -> Self {
        Self {
            registry,
            factories: HashMap::new(),
        }
    }

    /// Create an empty registry.
    pub fn empty() -> Self {
        Self {
            registry: HashMap::new(),
            factories: HashMap::new(),
        }
    }

    /// Register any Fn as a named function (stateful closures allowed).
    pub fn register<F>(&mut self, name: &'static str, f: F, doc: &'static str)
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        self.registry.insert(name, FuncMeta::new(f, doc));
    }

    /// Convenience for registering a plain function pointer.
    pub fn register_fn(&mut self, name: &'static str, f: fn(f32) -> f32, doc: &'static str) {
        self.registry.insert(name, FuncMeta::new(f, doc));
    }

    /// Register a factory under a name. Factories take a slice of f32 parameters and
    /// return an Arc<dyn Fn(f32)->f32> (a closure or function) built from those params.
    pub fn register_factory<F>(&mut self, name: &'static str, factory: F)
    where
        F: Fn(&[f32]) -> Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static> + Send + Sync + 'static,
    {
        self.factories.insert(name, Box::new(factory));
    }

    /// Call a registered factory with params. Returns Some(Arc<dyn Fn...>) or None if factory missing.
    pub fn call_factory(
        &self,
        name: &str,
        params: &[f32],
    ) -> Option<Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>> {
        self.factories.get(name).map(|f| f(params))
    }

    /// Remove a registered function. Returns true if an entry existed and was removed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.registry.remove(name).is_some()
    }

    /// Replace an existing registration or insert if missing. Returns the previous entry if any.
    pub fn replace<F>(&mut self, name: &'static str, f: F, doc: &'static str) -> Option<FuncMeta>
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        self.registry.insert(name, FuncMeta::new(f, doc))
    }

    pub fn list_functions(&self) {
        for (name, meta) in &self.registry {
            println!("{}: {}", name, meta.doc());
        }
    }

    /// Return the registered function as an Arc if present.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>> {
        self.registry.get(name).map(|m| m.function())
    }
}
