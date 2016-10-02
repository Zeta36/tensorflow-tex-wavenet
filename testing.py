import fnmatch
import os
import tensorflow as tf
import numpy as np

def find_files(directory, pattern='*.txt'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def _read_text(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return list(f.read().decode("utf-8").replace("\n", ""))

def load_generic_text(directory):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory)
    for filename in files:
        text = _read_text(filename)

        for index, item in enumerate(text):
            text[index] = ord(text[index])

        text = np.array(text, dtype='float32')
        print text
        text = text.reshape(-1, 1)

        print (text)
        y = []
        for index, item in enumerate(text):
            y.append(chr(text[index]))
            
        print(y)
        yield text, filename

        
def main():
    iterator = load_generic_text("./data")
    for text, filename in iterator:
        print ("Filename", filename)

if __name__ == '__main__':
    main()