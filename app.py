"""
app.py  –  Project entry point (replaces the original Flask app.py).

CLI usage (unchanged from original):
    python app.py <input_folder> <output_folder> <summary_lines>

API server:
    python app.py --web [--port 8000]
    or:
    uvicorn api.main:app --reload
"""

import sys


def run_cli() -> None:
    if len(sys.argv) < 4:
        print("Usage: python app.py <input_folder> <output_folder> <summary_lines>")
        print("  API: python app.py --web [--port 8000]")
        return

    from pipeline.batch_pipeline import BatchPipeline  # noqa
    BatchPipeline().process_folder(
        input_folder=sys.argv[1],
        output_folder=sys.argv[2],
        summary_lines=int(sys.argv[3]),
    )


if __name__ == "__main__":
    if "--web" in sys.argv:
        import uvicorn
        from api.main import app

        port = 8000
        if "--port" in sys.argv:
            idx = sys.argv.index("--port") + 1
            if idx < len(sys.argv):
                port = int(sys.argv[idx])

        uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
    else:
        run_cli()
