def split_file(input_path, lines_per_chunk=5000):
    """
    Splits a large text file into smaller chunk files.
    Each chunk contains `lines_per_chunk` lines.
    """
    chunk_index = 1
    current_lines = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            current_lines.append(line)

            if len(current_lines) >= lines_per_chunk:
                output_path = f"chunk_{chunk_index}.txt"
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.writelines(current_lines)
                current_lines = []
                chunk_index += 1

    # write remaining lines
    if current_lines:
        output_path = f"chunk_{chunk_index}.txt"
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(current_lines)

    print(f"Done. Created {chunk_index} chunk files.")

split_file("master_ranked_all_words.txt", lines_per_chunk=5000)