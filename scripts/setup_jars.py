import os
import time
import urllib.request
import sys

OLD_GUAVA = "guava-14.0.1.jar"
jars_dir = os.path.join(sys.prefix, "Lib", "site-packages", "pyspark", "jars")
jar_urls = {
    "hadoop-aws-3.2.1.jar": "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.1/hadoop-aws-3.2.1.jar",
    "aws-java-sdk-bundle-1.11.1026.jar": "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar",
    "guava-27.0-jre.jar": "https://repo1.maven.org/maven2/com/google/guava/guava/27.0-jre/guava-27.0-jre.jar"
}


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
    remove_old_guava(jars_dir)
    download_jars(jars_dir)
