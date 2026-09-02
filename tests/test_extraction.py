from screening.extraction import extract_profile


def test_name_extraction_handles_header_line_with_contact_info():
    text = "Yuvaraj R    9500647387    Thanjavur, Tamil Nadu\nEmail: yr@example.com\n"
    profile = extract_profile(text)
    assert profile.name == "Yuvaraj R"


def test_name_extraction_handles_one_word_per_line_template():
    text = "ABHINAV\n\nMISHRA\n\nGhaziabad,\n\nUP\n\n|\n\nabhinavmishra3322@gmail.com\n"
    profile = extract_profile(text)
    assert profile.name == "ABHINAV MISHRA"


def test_github_username_extracted_from_profile_url_not_repo_url():
    text = "Contact: github.com/octocat | octocat@example.com"
    profile = extract_profile(text)
    assert profile.github_username == "octocat"
    assert profile.github_url == "github.com/octocat"


def test_no_github_or_email_present_leaves_fields_none_without_raising():
    text = "Some Name\nSkills\nPython, Django\n"
    profile = extract_profile(text)
    assert profile.github_username is None
    assert profile.email is None
    assert profile.name == "Some Name"


def test_missing_fields_do_not_raise():
    profile = extract_profile("")
    assert profile.name is None
    assert profile.email is None
    assert profile.github_username is None
