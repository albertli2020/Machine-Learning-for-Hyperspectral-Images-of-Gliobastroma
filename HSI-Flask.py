from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import subprocess
import json
import os

app = Flask(__name__)

CONFIG_FILE = "config.json"
SCRIPTS_DIR = "./"

# Load configuration
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)
else:
    config_data = {}

# Extract available scripts and default selection
script_files = {}
for script, details in config_data.items():
    if isinstance(details, dict) and "script_file" in details:
        script_files[script] = details["script_file"]

# Load the initially selected script
selected_script = config_data.get("selected_script", None)

print("Loaded script files:", script_files)  # Debugging log
print("Initial selected script:", selected_script)  # Debugging log

@app.route('/')
def index():
    return render_template("index.html", scripts=script_files, config=config_data, selected_script=selected_script)

@app.route('/get_config', methods=['POST'])
def get_config():
    selected_script = request.json.get("script")
    config_data["selected_script"] = selected_script
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)
    script_config = config_data.get(selected_script, {})
    return jsonify(script_config)

@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    script = data.get("script")
    updated_config = data.get("config", {})

    # Debug: Check the updated configuration
    #print(f"Updating config for script {script}: {updated_config}")

    if script in config_data:
        for key, value in updated_config.items():
            print(key, value)
            if "selected" in value and key in config_data[script]:
                config_data[script][key]["selected"] = value["selected"]  # Update selected value
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

    return jsonify({"message": "Configuration updated"})

@app.route('/execute', methods=['GET'])#) #, methods=['POST'])
def execute_script():
    script_dec = request.args.get("script")
    script_name = script_files.get(script_dec)
    if not script_name:
        return jsonify({"error": "Invalid script selection"})
    
    try:        
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        @stream_with_context
        def async_results():
            print(f"Starting subprocess... {script_dec}")  # Debugging
            # Set the environment variable to disable buffering
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(["python", script_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env=env, text=True, universal_newlines=True)
            
            # Stream stdout line by line
            for line in iter(process.stdout.readline, ''):
                #print(f"Got stdout: {line.strip()}")
                # Create JSON object for the stdout line
                json_data = json.dumps({"output": line.strip(), "error": None})
                #print(f"Yielding: {json_data}")  # Optional logging for debugging
                yield f"{json_data}\n"

            process.stdout.close()  # Close stdout after reading

            # Handle any errors from stderr
            error_output = process.stderr.read().strip()
            if error_output:
                json_data = json.dumps({"output": None, "error": error_output})
                #print(f"Yielding: {json_data}")  # Optional logging for debugging
                yield f"{json_data}\n"

            process.wait()  # Ensure the process has finished                                                

            #result = subprocess.run(["python", os.path.join(SCRIPTS_DIR, script_name)], capture_output=True, text=True)
            #output_entry = f"{result.stdout or 'No output'}{'-'*40}"
        return Response(async_results(), content_type="application/json")
        
    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),
            content_type="application/json"  # Ensure error response also returns JSON
        )
        #return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
