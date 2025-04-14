import sys

def main(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                print("hello " + line.strip())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test.py <file_path>")
    else:
        main(sys.argv[1])
