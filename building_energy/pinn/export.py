"""
ONNX Export and NPU Adaptation Module

This module provides functionality to export trained PINN models to ONNX format
and adapt them for NPU (Neural Processing Unit) inference on edge devices.
Supported NPUs include Huawei Ascend (MindSpore) and Cambricon MLU (PaddlePaddle).
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
    logger.info(f"ONNX version: {onnx.__version__}")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
    logger.info(f"ONNX Runtime version: {ort.__version__}")
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

try:
    import torch
    import torch.onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mindspore as ms
    from mindspore import context, Tensor
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

try:
    import paddle
    import paddle.static as static
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class NPUAdapter:
    """
    Adapter for NPU (Neural Processing Unit) inference.
    
    Supports:
    - Huawei Ascend (via MindSpore)
    - Cambricon MLU (via PaddlePaddle)
    - Generic ONNX Runtime
    """
    
    def __init__(self, backend: str = "onnxruntime"):
        """
        Initialize NPU adapter.
        
        Args:
            backend: NPU backend ('ascend', 'mlu', 'onnxruntime')
        """
        self.backend = self._validate_backend(backend)
        self.session = None
        self.input_name = None
        self.output_name = None
        
        logger.info(f"NPUAdapter initialized with backend: {self.backend}")
    
    def _validate_backend(self, backend: str) -> str:
        """Validate and select available backend."""
        backend = backend.lower()
        
        if backend == "ascend":
            if not MINDSPORE_AVAILABLE:
                logger.warning("MindSpore not available, falling back to ONNX Runtime")
                backend = "onnxruntime"
            else:
                # Check if Ascend is available
                try:
                    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
                    logger.info("Ascend NPU detected")
                except Exception as e:
                    logger.warning(f"Ascend not available: {e}, falling back to CPU")
                    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        
        elif backend == "mlu":
            if not PADDLE_AVAILABLE:
                logger.warning("PaddlePaddle not available, falling back to ONNX Runtime")
                backend = "onnxruntime"
            else:
                # Check if MLU is available
                try:
                    paddle.set_device("mlu")
                    logger.info("Cambricon MLU detected")
                except Exception as e:
                    logger.warning(f"MLU not available: {e}, falling back to CPU")
                    paddle.set_device("cpu")
        
        elif backend == "onnxruntime":
            if not ONNXRUNTIME_AVAILABLE:
                raise ImportError("ONNX Runtime not available")
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        return backend
    
    def load_model(self, model_path: str):
        """
        Load a model for NPU inference.
        
        Args:
            model_path: Path to model file (.onnx, .mindir, or paddle format)
        """
        if self.backend == "onnxruntime":
            self._load_onnx_model(model_path)
        elif self.backend == "ascend":
            self._load_mindspore_model(model_path)
        elif self.backend == "mlu":
            self._load_paddle_model(model_path)
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        # Check for NPU providers
        providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {providers}")
        
        # Prioritize NPU providers
        preferred_providers = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'ROCMExecutionProvider',
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        selected_providers = [p for p in preferred_providers if p in providers]
        
        self.session = ort.InferenceSession(model_path, providers=selected_providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ONNX model loaded. Input: {self.input_name}, Output: {self.output_name}")
    
    def _load_mindspore_model(self, model_path: str):
        """Load MindSpore model for Ascend."""
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore not available")
        
        # Load MindIR model
        self.session = ms.load(model_path)
        logger.info("MindSpore model loaded")
    
    def _load_paddle_model(self, model_path: str):
        """Load PaddlePaddle model for MLU."""
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddlePaddle not available")
        
        # Load Paddle inference model
        # Assuming model_path is a directory with __model__ and __params__
        self.exe = static.Executor(paddle.CPUPlace())
        [self.inference_program, self.feed_target_names, self.fetch_targets] = (
            static.io.load_inference_model(model_path, self.exe)
        )
        logger.info("PaddlePaddle model loaded")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input array of shape [batch_size, 4] (x, y, z, t)
            
        Returns:
            Prediction array of shape [batch_size, 1] (temperature)
        """
        if self.backend == "onnxruntime":
            return self._onnx_predict(input_data)
        elif self.backend == "ascend":
            return self._mindspore_predict(input_data)
        elif self.backend == "mlu":
            return self._paddle_predict(input_data)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def _onnx_predict(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX Runtime prediction."""
        # Ensure float32
        input_data = input_data.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        return outputs[0]
    
    def _mindspore_predict(self, input_data: np.ndarray) -> np.ndarray:
        """MindSpore prediction."""
        input_tensor = Tensor(input_data, ms.float32)
        output = self.session(input_tensor)
        return output.asnumpy()
    
    def _paddle_predict(self, input_data: np.ndarray) -> np.ndarray:
        """PaddlePaddle prediction."""
        import paddle
        
        input_tensor = paddle.to_tensor(input_data, dtype='float32')
        results = self.exe.run(
            self.inference_program,
            feed={self.feed_target_names[0]: input_tensor},
            fetch_list=self.fetch_targets
        )
        return results[0]
    
    def benchmark(self, n_iterations: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            n_iterations: Number of iterations for benchmarking
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Warm up
        dummy_input = np.random.randn(batch_size, 4).astype(np.float32)
        for _ in range(10):
            _ = self.predict(dummy_input)
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.time()
            _ = self.predict(dummy_input)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        results = {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_qps": 1000.0 / np.mean(latencies) * batch_size
        }
        
        logger.info(f"Benchmark results: mean={results['mean_latency_ms']:.2f}ms, "
                   f"p95={results['p95_latency_ms']:.2f}ms")
        
        return results


class ONNXExporter:
    """
    Exporter for converting trained models to ONNX format.
    
    Supports exporting from PyTorch, MindSpore, and generic formats.
    """
    
    def __init__(self, input_shape: Tuple[int, ...] = (1, 4)):
        """
        Initialize exporter.
        
        Args:
            input_shape: Expected input shape (batch_size, features)
        """
        self.input_shape = input_shape
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx")
    
    def export_from_pytorch(
        self,
        model: Any,
        output_path: str,
        dummy_input: Optional[np.ndarray] = None
    ) -> str:
        """
        Export PyTorch model to ONNX.
        
        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            dummy_input: Dummy input for tracing
            
        Returns:
            Path to exported ONNX file
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if dummy_input is None:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        dummy_tensor = torch.from_numpy(dummy_input)
        
        # Export
        torch.onnx.export(
            model,
            dummy_tensor,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"PyTorch model exported to: {output_path}")
        
        # Verify
        self._verify_onnx(output_path, dummy_input)
        
        return output_path
    
    def export_from_mindspore(
        self,
        model: Any,
        output_path: str,
        file_format: str = "MINDIR"
    ) -> str:
        """
        Export MindSpore model.
        
        Args:
            model: MindSpore model
            output_path: Output file path
            file_format: Export format ('MINDIR' or 'ONNX')
            
        Returns:
            Path to exported file
        """
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore not available")
        
        if file_format.upper() == "MINDIR":
            # Export to MindIR format (MindSpore native)
            ms.export(model, output_path, file_format="MINDIR")
            logger.info(f"MindSpore model exported to: {output_path}")
        elif file_format.upper() == "ONNX":
            # Export to ONNX
            ms.export(model, output_path, file_format="ONNX")
            logger.info(f"MindSpore model exported to ONNX: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        return output_path
    
    def export_from_paddle(
        self,
        model: Any,
        output_path: str,
        dummy_input: Optional[np.ndarray] = None
    ) -> str:
        """
        Export PaddlePaddle model.
        
        Args:
            model: PaddlePaddle model
            output_path: Output directory path
            dummy_input: Dummy input for tracing
            
        Returns:
            Path to output directory
        """
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddlePaddle not available")
        
        import paddle.static as static
        
        if dummy_input is None:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Save inference model
        static.io.save_inference_model(
            output_path,
            ['input'],
            [model],
            static.Executor(paddle.CPUPlace())
        )
        
        logger.info(f"PaddlePaddle model exported to: {output_path}")
        
        return output_path
    
    def _verify_onnx(self, model_path: str, test_input: np.ndarray):
        """Verify ONNX model by running inference."""
        if not ONNXRUNTIME_AVAILABLE:
            logger.warning("ONNX Runtime not available, skipping verification")
            return
        
        # Load and test
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        test_input = test_input.astype(np.float32)
        outputs = session.run(None, {input_name: test_input})
        
        logger.info(f"ONNX verification successful. Output shape: {outputs[0].shape}")
    
    def optimize_onnx(
        self,
        input_path: str,
        output_path: str,
        optimizations: Optional[List[str]] = None
    ) -> str:
        """
        Optimize ONNX model for inference.
        
        Args:
            input_path: Input ONNX file path
            output_path: Output optimized ONNX file path
            optimizations: List of optimization passes
            
        Returns:
            Path to optimized model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        # Load model
        model = onnx.load(input_path)
        
        # Check model
        onnx.checker.check_model(model)
        
        # Apply optimizations
        if optimizations is None:
            optimizations = [
                "eliminate_identity",
                "fuse_consecutive_transposes",
                "fuse_pad_into_conv",
                "extract_constant_to_initializer",
                "fuse_bn_into_conv"
            ]
        
        # Use ONNX optimizer if available
        try:
            from onnx import optimizer
            optimized_model = optimizer.optimize(model, optimizations)
            onnx.save(optimized_model, output_path)
            logger.info(f"ONNX model optimized and saved to: {output_path}")
        except ImportError:
            # onnx.optimizer was removed in newer versions
            # Use onnxruntime tools instead
            logger.warning("ONNX optimizer not available, saving original model")
            onnx.save(model, output_path)
        
        return output_path


def export_and_benchmark(
    model_path: str,
    output_dir: str = "models/pinn",
    npu_backend: str = "onnxruntime"
) -> Dict[str, Any]:
    """
    Export model and benchmark on NPU.
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory
        npu_backend: NPU backend to use
        
    Returns:
        Dictionary with export and benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "export_success": False,
        "benchmark": None,
        "model_path": None
    }
    
    # Export to ONNX
    try:
        exporter = ONNXExporter()
        onnx_path = os.path.join(output_dir, "thermal_model.onnx")
        
        # Assuming model_path is a checkpoint, we'd need to load it first
        # For now, just verify the ONNX file exists
        if os.path.exists(onnx_path):
            results["export_success"] = True
            results["model_path"] = onnx_path
            logger.info(f"ONNX model found at: {onnx_path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return results
    
    # Benchmark on NPU
    try:
        adapter = NPUAdapter(backend=npu_backend)
        adapter.load_model(onnx_path)
        benchmark_results = adapter.benchmark(n_iterations=100)
        results["benchmark"] = benchmark_results
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    
    return results
