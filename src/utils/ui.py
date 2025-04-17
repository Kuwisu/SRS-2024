def get_article(value):
    """ Returns an indefinite article preceding an integer, either 'a' or 'an' """
    first_digit = str(value)[0]
    # 11 and any value starting with 8 has an 'e' rather than a consonant sound
    if first_digit == '8' or value == 11:
        return 'an'
    return 'a'

def retrieve_int_field(line_edit, is_valid=True):
    """
    Text entries are checked to make sure they are valid integers.
    If not, the text box is cleared and coloured red to signal an error.

    :param line_edit: the text field to validate
    :param is_valid: whether the overall form is currently valid
    :return: True if integer or disabled, False otherwise; also return
    the text contained in the line edit
    """
    try:
        value = int(line_edit.text())
    except ValueError:
        value = 0
        if line_edit.isEnabled():
            line_edit.setText("")
            line_edit.setStyleSheet("border: 1px solid red;")
            return False, value
    return is_valid, value
