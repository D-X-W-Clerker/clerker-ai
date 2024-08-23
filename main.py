import os
import subprocess
import webbrowser
import json

def run_inference(folder_name, input_data):
    command = ["python", f"{folder_name}/inference.py", input_data]
    result = subprocess.run(command, shell=False, check=True, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def extract_mermaid_code(full_text):
    start_tag = "```mermaid"
    end_tag = "```"
    
    start_index = full_text.find(start_tag)
    if start_index != -1:
        start_index += len(start_tag)
        start_index = full_text.find("\n", start_index) + 1
        
        end_index = full_text.find(end_tag, start_index)
        
        if start_index != -1 and end_index != -1:
            mermaid_code = full_text[start_index:end_index].strip()
            return mermaid_code

    return None

def create_html_file(content_list):
    html_template_start = """
    <!DOCTYPE html>
    <html lang="en">
      <body>
    """

    html_template_end = """
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        </script>
      </body>
    </html>
    """

    content_html = ""
    for idx, content in enumerate(content_list):
        content_html += f"""
        <h2>Chunk {idx + 1}</h2>
        <pre class="mermaid">
{content}
        </pre>
        """

    full_html = html_template_start + content_html + html_template_end

    output_file = "output.html"
    with open(output_file, "w") as file:
        file.write(full_html)

    webbrowser.open(f"file://{os.path.abspath(output_file)}")

def main(audio_file):
    print('complete-audio')
    print(audio_file)
    stt_result = run_inference("stt", audio_file)
    print('complete-stt')
    
    summary_result = run_inference("summarization", stt_result)
    print('complete-summarization')

    summary_json_path = os.path.abspath('./summarized_text.json')
    with open(summary_json_path, "r", encoding="utf-8") as file:
        summary_data = json.load(file)
    
    mermaid_codes = []
    
    for chunk in summary_data["chunks"]:
        print(chunk)
        instruction_text = (
            f"{chunk['summarized_text']}\n\n"
            "위처럼 요약한 회의 내용을 바탕으로 다이어그램을 생성할거야. "
            "아래의 다이어그램 종류 중 회의 내용에 가장 적합한 다이어그램으로 mermaid 코드 만들어줘.\n\n"
            "Flowchart, Sequence Diagram, Class Diagram, State Diagram, Entity Relationship Diagram, "
            "User Journey, Gantt, Pie Chart, Quadrant Chart, Requirement Diagram, Gitgraph (Git) Diagram, "
            "C4 Diagram, Mindmaps, Timeline, Zenuml, Sankey, XYChart, Block Diagram"
        )

        full_text = run_inference("sllm", instruction_text)
        print(f'complete-mermaid for chunk {chunk["chunk_num"]}')
        
        mermaid_code = extract_mermaid_code(full_text)
        if mermaid_code:
            mermaid_codes.append(mermaid_code)
    
    create_html_file(mermaid_codes)
    print('create-html')

if __name__ == "__main__":
    raw_audio = "./audio_file/example.mp3"  
    main(raw_audio)
