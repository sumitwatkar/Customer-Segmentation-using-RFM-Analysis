import sys

# Define a custom exception class
class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):  # Constructor method
        # Call method to get detailed error message and assign it to instance variable
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,
                                                                         error_details=error_details)
        
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:  # Static method to get detailed error message
        _, _, exce_tb = error_details.exc_info()  # Get the exception traceback
        
        # Extract line numbers and file name from traceback
        exception_block_line_number = exce_tb.tb_frame.f_lineno
        try_block_line_number = exce_tb.tb_lineno
        file_name = exce_tb.tb_frame.f_code.co_filename

        # Construct detailed error message
        error_message = f"""
        Error occurred in execution of:
        [{file_name}] at
        try block line number: [{try_block_line_number}]
        and exception block line number: [{exception_block_line_number}]
        error message: [{error_message}]
        """
        return error_message
    
    def __str__(self):  # Override __str__ method to return error message as string
        return self.error_message
    
    def __repr__(self):  # Override __repr__ method to return class name as string
        return CustomException.__name__.str()