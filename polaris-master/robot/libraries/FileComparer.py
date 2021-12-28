"""
File Comparer module
"""
import os
from filecmp import cmp

from robot.api.deco import keyword, library


@library
class FileComparer:
    """ File Comparer predictor class
    """
    @keyword
    def compare_files(self, first_file_path, second_file_path):
        """ Test if two files are identicals

            :param first_file_path: Path to the first file
            :type first_file_path: String
            :param second_file_path: Path to the second file
            :type second_file_path: String
            :return: The result of the comparison
            :rtype: Boolean
        """
        return cmp(first_file_path, second_file_path, shallow=False)

    @keyword
    def compare_files_sizes(self, first_file_path, second_file_path):
        """ Test if two files sizes are identicals

            :param first_file_path: Path to the first file
            :type first_file_path: String
            :param second_file_path: Path to the second file
            :type second_file_path: String
            :return: The result of the size comparison
            :rtype: Boolean
        """
        return os.path.getsize(first_file_path) == os.path.getsize(
            second_file_path)
