import re

def process_file(in_path, out_path, title):
    with open(in_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove extends and block definition headers
    content = re.sub(r'\{%\s*extends\s+[^%]+\s*%\}', '', content)
    content = re.sub(r'\{%\s*block\s+[^\s]+\s*%\}', '', content)
    content = re.sub(r'\{%\s*endblock\s*%\}', '', content)

    # Clean up empty lines created by removing jinja blocks
    lines = []
    has_body = False
    for line in content.split('\n'):
        if line.strip() == '':
            continue
        lines.append(line)

    # Wrap in HTML
    html_skeleton = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
"""
    
    # Simple heuristic to wrap body
    html_content = html_skeleton
    in_css = False
    
    for line in lines:
        if "<style" in line:
            in_css = True
        
        # When we hit the first HTML element after styles
        if not has_body and not in_css and "<div" in line:
            html_content += "</head>\n<body>\n"
            has_body = True
        elif not has_body and in_css and "</style>" in line:
            in_css = False
            html_content += line + "\n</head>\n<body>\n"
            has_body = True
            continue
            
        html_content += line + "\n"

    html_content += "</body>\n</html>\n"

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Fixed {out_path}")

base_dir = "E:/02_StayG/00_CCTV_Motion_Detection"
original_monitor = base_dir + "/vlm_gravity/vlm_pipipeline/cctv/model_examine/vlm_pipeline/templates/vlm_pipeline/monitor_shadow.html"
original_adhoc = base_dir + "/vlm_gravity/vlm_pipipeline/cctv/model_examine/vlm_pipeline/templates/vlm_pipeline/adhoc_rtsp.html"

out_monitor = base_dir + "/vlm_gravity/vlm_new_deploy/frontend_server/templates/vlm_pipeline/monitor_shadow.html"
out_adhoc = base_dir + "/vlm_gravity/vlm_new_deploy/frontend_server/templates/vlm_pipeline/adhoc_rtsp.html"

process_file(original_monitor, out_monitor, "VLM Shadow Monitor")
process_file(original_adhoc, out_adhoc, "VLM RTSP 멀티 모니터")
