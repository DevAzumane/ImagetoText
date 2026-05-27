import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from engine.vlm_engine import QwenVLEngine

class BatchPipeline:

    def __init__(self):
        self.engine = QwenVLEngine()

    def _run_step_with_progress(
        self,
        task,
        completed_steps,
        total_steps,
        progress_callback,
        message,
        timeout_seconds=None,
    ):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(task)
        start = time.monotonic()

        try:
            while True:
                try:
                    result = future.result(timeout=2)
                    executor.shutdown(wait=False, cancel_futures=True)
                    return result
                except TimeoutError:
                    elapsed = int(time.monotonic() - start)

                    if timeout_seconds and elapsed >= timeout_seconds:
                        raise TimeoutError(
                            f"{message} timed out after {timeout_seconds} seconds. "
                            "Check that Ollama is running and the vision model is loaded."
                        )

                    if progress_callback:
                        if timeout_seconds:
                            step_fraction = min(
                                0.85,
                                elapsed / max(1, timeout_seconds),
                            )
                        else:
                            step_fraction = min(
                                0.85,
                                elapsed / (elapsed + 60),
                            )
                        progress_callback(
                            completed_steps + step_fraction,
                            total_steps,
                            f"{message}... {elapsed}s",
                        )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    

    def process_folder(
        self,
        input_folder,
        output_folder,
        summary_lines=3,
        progress_callback=None,
    ):

        os.makedirs(output_folder, exist_ok=True)

        image_extensions = [".png", ".jpg", ".jpeg"]

        files = [
            f for f in os.listdir(input_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        files = sorted(files)
        total_steps = max(1, len(files) * 3)
        completed_steps = 0

        for file_name in files:

            image_path = os.path.join(input_folder, file_name)

            print(f"\n[PROCESSING] {file_name}")

            # STEP 1: OCR
            if progress_callback:
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"Reading text from {file_name}",
                )

            extracted_text = self._run_step_with_progress(
                task=lambda: self.engine.extract_text(image_path),
                completed_steps=completed_steps,
                total_steps=total_steps,
                progress_callback=progress_callback,
                message=f"Reading text from {file_name}",
            )
            completed_steps += 1
            if progress_callback:
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"OCR complete for {file_name}",
                )

            print("[OCR COMPLETE]")

            # STEP 2: SUMMARY
            summary = self._run_step_with_progress(
                task=lambda: self.engine.summarize_text(
                    extracted_text,
                    num_lines=summary_lines,
                ),
                completed_steps=completed_steps,
                total_steps=total_steps,
                progress_callback=progress_callback,
                message=f"Summarizing {file_name}",
            )
            completed_steps += 1
            if progress_callback:
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"Summary complete for {file_name}",
                )

            print("[SUMMARY COMPLETE]")

            # STEP 3: SAVE OUTPUT
            output_file = os.path.join(
                output_folder,
                f"{os.path.splitext(file_name)[0]}.txt"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("===== EXTRACTED TEXT =====\n\n")
                f.write(extracted_text)
                f.write("\n\n")
                f.write("===== SUMMARY =====\n\n")
                f.write(summary)

            print(f"[SAVED] {output_file}")            
            completed_steps += 1
            if progress_callback:
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"Saved summary for {file_name}",
                )
