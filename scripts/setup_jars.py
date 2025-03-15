import os
import time
import urllib.request
import sys
import subprocess


OLD_GUAVA = "guava-14.0.1.jar"
jars_dir = os.path.join(sys.prefix, "Lib", "site-packages", "pyspark", "jars")
jar_urls = {
    "hadoop-aws-3.2.1.jar": "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.1/hadoop-aws-3.2.1.jar",
    "aws-java-sdk-bundle-1.11.1026.jar": "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar",
    "guava-27.0-jre.jar": "https://repo1.maven.org/maven2/com/google/guava/guava/27.0-jre/guava-27.0-jre.jar",
    "hadoop-common-3.2.1.jar": "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.2.1/hadoop-common-3.2.1.jar",
    "hadoop-client-3.2.1.jar": "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client/3.2.1/hadoop-client-3.2.1.jar"
}
HADOOP_VERSION = "3.3.6"
HADOOP_HOME_DEFAULT = os.path.join(os.getcwd(), f"hadoop-{HADOOP_VERSION}")
WINUTILS_URL = f"https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin/winutils.exe"

def check_environment():
    """Kiểm tra môi trường hiện tại"""
    print("🔍 Kiểm tra môi trường:")
    print(f"- Hệ điều hành: {os.name}")  # 'posix' cho Linux, 'nt' cho Windows
    print(f"- Thư mục hiện tại: {os.getcwd()}")
    
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and os.path.exists(hadoop_home):
        print(f"✅ HADOOP_HOME hiện tại: {hadoop_home}")
    else:
        print(f"⚠️ HADOOP_HOME chưa được thiết lập")
        
def is_hadoop_installed():
    """Kiểm tra xem Hadoop đã được cài trên Windows chưa"""
    try:
        result = subprocess.run(["where", "hadoop"], capture_output=True, text=True, shell=True)
        return result.returncode == 0
    except Exception:
        return False


def setup_hadoop_home():
    """Thiết lập HADOOP_HOME và tải winutils.exe nếu cần"""
    if os.name != "nt":  # Không cần thiết lập trên Linux
        print("Không cần thiết lập HADOOP_HOME trên hệ điều hành không phải Windows.")
        return

    # Kiểm tra nếu Hadoop đã được cài đặt trước đó
    if is_hadoop_installed():
        print("✅ Đã phát hiện Hadoop được cài đặt trên hệ thống.")
        return

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
        print(f"✅ HADOOP_HOME đã được thiết lập tại: {hadoop_home}")
        return

    # Thiết lập HADOOP_HOME mặc định nếu chưa có
    print(f"⚠️ HADOOP_HOME chưa được thiết lập. Thiết lập mặc định tại: {HADOOP_HOME_DEFAULT}")
    os.makedirs(HADOOP_HOME_DEFAULT, exist_ok=True)
    bin_dir = os.path.join(HADOOP_HOME_DEFAULT, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    # Tải winutils.exe
    winutils_path = os.path.join(bin_dir, "winutils.exe")
    if not os.path.exists(winutils_path):
        print(f"📥 Tải winutils.exe cho Hadoop {HADOOP_VERSION}...")
        urllib.request.urlretrieve(WINUTILS_URL, winutils_path)
        print(f"✅ Đã tải winutils.exe vào {winutils_path}")
    else:
        print(f"✅ winutils.exe đã tồn tại tại {winutils_path}")

    # Cập nhật biến môi trường trong session hiện tại
    os.environ["HADOOP_HOME"] = HADOOP_HOME_DEFAULT
    os.environ["PATH"] = f"{HADOOP_HOME_DEFAULT}\\bin;{os.environ['PATH']}"
    
    # Thiết lập biến môi trường vĩnh viễn trong Windows
    subprocess.run(["setx", "HADOOP_HOME", HADOOP_HOME_DEFAULT], shell=True)
    subprocess.run(["setx", "PATH", f"{HADOOP_HOME_DEFAULT}\\bin;%PATH%"], shell=True)

    print(f"✅ Đã thiết lập HADOOP_HOME vĩnh viễn = {HADOOP_HOME_DEFAULT}")
    print(f"⚠️ Bạn cần **khởi động lại terminal** để biến môi trường có hiệu lực!")
    
def remove_old_guava(jars_dir):
    old_guava_path = os.path.join(jars_dir, OLD_GUAVA)
    if os.path.exists(old_guava_path):
        print(f"Found {OLD_GUAVA}, removing it...")
        os.remove(old_guava_path)
        if not os.path.exists(old_guava_path):
            print(f"Successfully removed {OLD_GUAVA}.")
        else:
            print(f"Failed to remove {OLD_GUAVA}.")
            sys.exit(1)
    else:
        print(f"{OLD_GUAVA} not found in {jars_dir}.")


def download_jars(jars_dir):
    os.makedirs(jars_dir, exist_ok=True)
    for jar_name, url in jar_urls.items():
        jar_path = os.path.join(jars_dir, jar_name)
        if not os.path.exists(jar_path):
            print(f"Downloading {jar_name}...")
            urllib.request.urlretrieve(url, jar_path)
            print(f"Downloaded {jar_name} to {jar_path}")
        else:
            print(f"{jar_name} already exists at {jar_path}")
        
        time.sleep(1)

if __name__ == "__main__":
    # Kiểm tra môi trường
    check_environment()
    
    # Thiết lập HADOOP_HOME và winutils.exe
    setup_hadoop_home()
    
    # Xóa guava cũ và tải các JAR
    remove_old_guava(jars_dir)
    download_jars(jars_dir)
    
    # Xác nhận lại môi trường sau khi thiết lập
    print("\nXác nhận môi trường sau khi thiết lập:")
    check_environment()