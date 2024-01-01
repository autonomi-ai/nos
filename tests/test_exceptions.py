from nos import exceptions as exc


def test_server_exception():
    message = "Test ServerException"
    exception = ValueError("Test Exception")
    server_exception = exc.ServerException(message, exception)

    assert server_exception.message.startswith(message)
    assert server_exception.exc == exception

    expected_error_message = f"{message}, details={exception}"
    assert str(server_exception) == expected_error_message


def test_model_not_found_error():
    message = "Test ModelNotFoundError"
    exception = ValueError("Test Exception")
    model_not_found_error = exc.ModelNotFoundError(message, exception)

    assert model_not_found_error.message.startswith(message)
    assert model_not_found_error.exc == exception

    expected_error_message = f"{message}, details={exception}"
    assert str(model_not_found_error) == expected_error_message
