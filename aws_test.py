import requests

def is_running_on_aws():
    try:
        # AWS metadata endpoint
        metadata_url = "http://169.254.169.254/latest/meta-data/"
        
        # Query the metadata to check if the endpoint is accessible
        response = requests.get(metadata_url, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if __name__ == "__main__":
    if is_running_on_aws():
        print("This script is running on an AWS server.")
    else:
        print("This script is not running on an AWS server.")
