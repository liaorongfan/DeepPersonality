def replace(string):
    replace_ch = ["{", "'O'", ",", "'C'", "'E'", "'A'", "'N'", "mean", "}"]
    for ch in replace_ch:
        string = string.replace(ch, "")
    string = string.replace(":", "&")
    return string

