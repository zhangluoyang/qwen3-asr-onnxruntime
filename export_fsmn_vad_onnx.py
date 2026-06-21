from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


class FsmnVadOnnxExportWrapper(torch.nn.Module):
    """Flattens FunASR export outputs for torch.onnx.export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, speech, in_cache0, in_cache1, in_cache2, in_cache3):
        logits, caches = self.model(speech, in_cache0, in_cache1, in_cache2, in_cache3)
        return (logits, caches[0], caches[1], caches[2], caches[3])


class FsmnVadOnnxExporter:
    """Exports local FunASR FSMN-VAD encoder to ONNX."""

    def __init__(
        self,
        model_dir: str | Path = "./fsmn",
        output_dir: str | Path = "./onnx_fsmn",
        output_name: str = "fsmn_vad_encoder.onnx",
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.device = device

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_name

    def export(
        self,
        opset_version: int = 13,
        dummy_frames: int = 30,
        copy_assets: bool = True,
    ) -> Path:
        from funasr import AutoModel

        self.output_dir.mkdir(parents=True, exist_ok=True)
        automodel = AutoModel(
            model=str(self.model_dir),
            device=self.device,
            disable_update=True,
            disable_pbar=True,
        )
        model = automodel.model.export(
            type="onnx",
            encoder=automodel.kwargs.get("encoder", "FSMN"),
        )
        model.eval()
        wrapper = FsmnVadOnnxExportWrapper(model).eval()
        dummy_inputs = model.export_dummy_inputs(frame=int(dummy_frames))

        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(self.output_path),
            input_names=model.export_input_names(),
            output_names=model.export_output_names(),
            dynamic_axes=model.export_dynamic_axes(),
            opset_version=int(opset_version),
            do_constant_folding=True,
        )

        if copy_assets:
            self._copy_assets()
        return self.output_path

    def _copy_assets(self) -> None:
        for name in ("config.yaml", "configuration.json", "am.mvn"):
            src = self.model_dir / name
            if src.exists():
                shutil.copy2(src, self.output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FSMN-VAD encoder to ONNX.")
    parser.add_argument("--model-dir", default=Path("./fsmn"), type=Path)
    parser.add_argument("--output-dir", default=Path("./onnx_fsmn"), type=Path)
    parser.add_argument("--output-name", default="fsmn_vad_encoder.onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset-version", default=13, type=int)
    parser.add_argument("--dummy-frames", default=30, type=int)
    parser.add_argument("--no-copy-assets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = FsmnVadOnnxExporter(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        device=args.device,
    )
    output_path = exporter.export(
        opset_version=args.opset_version,
        dummy_frames=args.dummy_frames,
        copy_assets=not args.no_copy_assets,
    )
    print(f"exported: {output_path}")


if __name__ == "__main__":
    main()
