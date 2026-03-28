import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PATH = os.path.join(BASE_DIR, "for_user_validation")

def run_validation():
    # 1. Ensure directory exists
    if not os.path.exists(DIR_PATH):
        print(f"--- Error: Directory '{DIR_PATH}' not found. ---")
        return

    # 2. Get list of files
    files = [f for f in os.listdir(DIR_PATH) if f.endswith(".json")]

    if not files:
        print("--- No movies pending validation. ---")
        return

    print(f"--- Found {len(files)} movies to review ---\n")

    # 3. Process each file
    for file_name in files:
        file_path = os.path.join(DIR_PATH, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                movie_data = json.load(f)

            # Display the data
            print("=" * 50)
            print(f"FILE: {file_name}")
            print("-" * 50)
            print(json.dumps(movie_data, indent=4, ensure_ascii=False))
            print("-" * 50)

            # 4. User Interaction Loop
            while True:
                prompt = "Choose: [a]ccept, [d]iscard, [s]kip, [q]uit: "
                user_input = input(prompt).lower().strip()

                if user_input == 'a':
                    print(f">> ACCEPTED: {movie_data.get('title')} (Logic TBD)")
                    break
                elif user_input == 'd':
                    print(f">> DISCARDED: {movie_data.get('title')} (Logic TBD)")
                    break
                elif user_input == 's':
                    print(">> Skipping to next...")
                    break
                elif user_input == 'q':
                    print("Exiting...")
                    return
                else:
                    print("Invalid input. Please enter 'a', 'd', 's', or 'q'.")

            print("\n")

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

    print("--- Review session complete. ---")


if __name__ == "__main__":
    run_validation()