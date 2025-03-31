import os
import sys
from pathlib import Path
import winreg

def find_hadoop_home() -> str:
    """Try multiple ways to find HADOOP_HOME"""

    # 1. Try environment variable first
    hadoop_home = os.environ.get('HADOOP_HOME')
    if hadoop_home and Path(hadoop_home).exists():
        print("Found at environment variable")
        return hadoop_home

    # 2. Try Windows Registry with path validation
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
            hadoop_home = winreg.QueryValueEx(key, "HADOOP_HOME")[0]
            # Check if path contains 'hadoop' and exists
            if hadoop_home and 'hadoop' in hadoop_home.lower():
                hadoop_path = Path(hadoop_home)
                if hadoop_path.exists() and hadoop_path.is_dir():
                    # Verify it's actually a Hadoop directory
                    bin_dir = hadoop_path / 'bin'
                    if bin_dir.exists() and (bin_dir / 'hadoop.cmd').exists():
                        print("Found valid Hadoop installation in Registry:", hadoop_home)
                        return str(hadoop_path)

    except WindowsError as e:
        print(f"Registry search failed: {e}")
    except Exception as e:
        print(f"Error validating Registry path: {e}")

    # 3. Try common installation paths with different Hadoop versions
    hadoop_versions = ['3.3.6', '3.3.5', '3.3.4', '3.2.0']
    common_paths = []

    # Add version-specific paths
    for version in hadoop_versions:
        common_paths.extend([
            rf"C:\hadoop\hadoop-{version}",
            rf"C:\Program Files\Hadoop\hadoop-{version}",
            rf"C:\Program Files (x86)\Hadoop\hadoop-{version}",
            os.path.join(os.path.expanduser("~"), f"hadoop-{version}")
        ])

    for path in common_paths:
        if Path(path).exists():
            print("Found at version-specific paths")
            return str(Path(path))

    return None

def setup_environment():
    # Find Hadoop installation
    hadoop_home = find_hadoop_home()

    if not hadoop_home:
        print("Warning: Could not find HADOOP_HOME in system")
        print("Falling back to local hadoop directory...")
        hadoop_home = str(Path(__file__).parent / 'hadoop')

        if not Path(hadoop_home).exists():
            print("Error: Local Hadoop directory not found!")
            print("Please either:")
            print("1. Set HADOOP_HOME environment variable")
            print("2. Install Hadoop in a standard location")
            print("3. Extract Hadoop binaries to:", hadoop_home)
            sys.exit(1)

    # Set environment variables
    os.environ['HADOOP_HOME'] = hadoop_home
    hadoop_bin = os.path.join(hadoop_home, 'bin')
    os.environ['PATH'] = f"{hadoop_bin}{os.pathsep}{os.environ['PATH']}"

    # Verify winutils.exe exists
    winutils_path = os.path.join(hadoop_bin, 'winutils.exe')
    if not os.path.exists(winutils_path):
        print(f"Warning: winutils.exe not found at {winutils_path}")
        print("Some Hadoop operations may fail")

    print(f"Using HADOOP_HOME: {hadoop_home}")
    return hadoop_home

if __name__ == "__main__":
    hadoop_home = setup_environment()

    # Force reload environment variables
    os.environ.update()

    # Import và chạy chương trình chính
    from main import main

    main()
