import os
import platform
import subprocess
import sys

def is_admin():
    try:
        return os.getuid() == 0  # Linux/Mac: Check if UID is 0 (root)
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # Windows: Check admin privileges

# Hàm cập nhật file hosts trên Windows
def update_hosts_file():
    os_name = platform.system()
    if os_name == "Windows":
        hosts_file = r"C:\Windows\System32\drivers\etc\hosts" # Windows hosts file path
    else:  # Linux or Mac
        hosts_file = "/etc/hosts"

    new_entry = "127.0.0.1   minio"

    # Check if the entry already exists
    if os.path.exists(hosts_file):
        with open(hosts_file, 'r') as f:
            if new_entry in f.read():
                print(f"Entry '{new_entry}' already exists in hosts file.")
                return

    # If not running with admin/root privileges, request elevation
    if not is_admin():
        print(f"Requires {'admin' if os_name == 'Windows' else 'sudo'} privileges to update hosts file...")
        if os_name == "Windows":
            # Re-run script with admin privileges on Windows
            subprocess.call(['powershell', '-Command', f"Start-Process python -ArgumentList '\"{sys.argv[0]}\"' -Verb RunAs"])
        else:  # Linux/Mac
            # Re-run script with sudo on Linux/Mac
            subprocess.call(['sudo', sys.executable] + sys.argv)
        sys.exit()

    # Append the new entry to the hosts file
    with open(hosts_file, 'a') as f:
        f.write(f"\n{new_entry}\n")
    print(f"Added '{new_entry}' to hosts file.")


update_hosts_file()
