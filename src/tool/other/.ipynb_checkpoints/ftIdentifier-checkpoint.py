import subprocess

def get_file_type(file_path):
    result = subprocess.run(["file", file_path], capture_output=True, text=True)
    return result.stdout.strip()

file_path = "src/main/traffic_data/pcap/202201/australiaeast/20220101/azure-australiaeast-01_pcap_00001_20220101211234"
file_type = get_file_type(file_path)
print(f"File Type: {file_type}")