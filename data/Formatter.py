import subprocess
import sys
import re

def solve_via_execution(api_response: str) -> str:
    # Search for code blocks in the response, specifically looking for Python code wrapped in triple backticks
    match = re.search(r"```(?:python)?\n(.*?)\n```", api_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        clean_code = match.group(1).strip()
    else:
        # Try to execute the whole response if no code block is found, maybe the model forgot to wrap it in a code block
        clean_code = api_response.strip() 

    try:
        result = subprocess.run(
            [sys.executable, "-c", clean_code],
            capture_output=True,
            text=True,
            timeout=3,
            check=True
        )
        return result.stdout.strip()
        
    except subprocess.CalledProcessError:
        return "EXECUTION_ERROR"
    except Exception:
        return "UNKNOWN_ERROR"

def clean_answer(answer: str) -> str:
    if answer is None:
        return ""

    cleaned = answer.strip()

    # Remove code block markers if present
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    return cleaned.strip()