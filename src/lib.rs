pub mod turbulance;

pub use turbulance::{
    TurbulanceCompiler, 
    CompilationTarget, 
    CompilationResult,
    ValidationResult,
    utils
};

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Initialize the Turbulance library
pub fn init() -> anyhow::Result<()> {
    env_logger::try_init().ok();
    Ok(())
}

/// Get the version of the Turbulance compiler
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn turbulance_compiler(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Python wrapper for TurbulanceCompiler
    #[pyclass]
    struct PyTurbulanceCompiler {
        compiler: TurbulanceCompiler,
    }

    #[pymethods]
    impl PyTurbulanceCompiler {
        #[new]
        fn new(output_dir: String) -> Self {
            Self {
                compiler: TurbulanceCompiler::new(output_dir),
            }
        }

        /// Compile a Turbulance source file
        fn compile_file(&self, source_path: String) -> PyResult<String> {
            match self.compiler.compile_file(&source_path) {
                Ok(result) => Ok(result.source_code),
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Compilation error: {}", e))),
            }
        }

        /// Validate Turbulance source code
        fn validate_source(&self, source: String) -> PyResult<bool> {
            match self.compiler.validate_source(&source) {
                Ok(result) => Ok(result.is_valid),
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Validation error: {}", e))),
            }
        }

        /// Get compiler version
        #[staticmethod]
        fn version() -> String {
            TurbulanceCompiler::version().to_string()
        }

        /// Get supported features
        #[staticmethod]
        fn supported_features() -> Vec<String> {
            TurbulanceCompiler::supported_features().iter().map(|s| s.to_string()).collect()
        }
    }

    m.add_class::<PyTurbulanceCompiler>()?;
    m.add_function(wrap_pyfunction!(py_compile_turbulance, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_turbulance, m)?)?;
    Ok(())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn py_compile_turbulance(source: String, output_dir: String) -> PyResult<String> {
    let compiler = TurbulanceCompiler::new(output_dir);
    match compiler.compile_string(&source, "turbulance_program", std::time::Instant::now()) {
        Ok(result) => Ok(result.source_code),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Compilation error: {}", e))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn py_validate_turbulance(source: String) -> PyResult<bool> {
    let compiler = TurbulanceCompiler::new("/tmp".to_string());
    match compiler.validate_source(&source) {
        Ok(result) => Ok(result.is_valid),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Validation error: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = TurbulanceCompiler::new("/tmp".to_string());
        assert_eq!(compiler.supported_features().len() > 0, true);
    }
} 