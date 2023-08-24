import os
import contextlib
import logging
import tarfile
import zipfile
import os.path as op
from os.path import join as pjoin
from hashlib import md5
from shutil import copyfileobj
from tqdm.auto import tqdm
from urllib.request import urlopen

ANURA_PUBLIC_URL = 'https://chorus.blob.core.windows.net/public/'

# Set a user-writeable file-system location to put files:
home = pjoin(os.getcwd(), 'datasets')


class FetcherError(Exception):
    pass


def _log(msg):
    """Helper function used as short hand for logging.
    """
    logger = logging.getLogger(__name__)
    logger.info(msg)


def copyfileobj_withprogress(fsrc, fdst, total_length, length=16 * 1024):
    for ii in tqdm(range(0, int(total_length), length), unit=" MB"):
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


def _already_there_msg(folder):
    """
    Prints a message indicating that a certain data-set is already in place
    """
    msg = 'Dataset is already in place. If you want to fetch it again '
    msg += 'please first remove the folder %s ' % folder
    _log(msg)


def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5=None):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5

    Parameters
    ----------
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against. If None (default), checking is
        skipped
    """
    if stored_md5 is not None:
        computed_md5 = _get_file_md5(filename)
        if stored_md5 != computed_md5:
            msg = """The downloaded file, %s, does not have the expected md5
   checksum of "%s". Instead, the md5 checksum was: "%s". This could mean that
   something is wrong with the file or that the upstream file has been updated.
   You can try downloading the file again or updating to the newest version of
   dipy.""" % (filename, stored_md5,
               computed_md5)
            raise FetcherError(msg)


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        try:
            response_size = opener.headers['content-length']
        except KeyError:
            response_size = None

        with open(fname, 'wb') as data:
            if response_size is None:
                copyfileobj(opener, data)
            else:
                copyfileobj_withprogress(opener, data, response_size)


def fetch_data(files, folder, data_size=None):
    """Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, md5). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the md5 checksum does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.
    Raises
    ------
    FetcherError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.

    """
    if not op.exists(folder):
        _log("Creating new folder %s" % folder)
        os.makedirs(folder)

    if data_size is not None:
        _log('Data size is approximately %s' % data_size)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if op.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        _log('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        check_md5(fullpath, md5)
    if all_skip:
        _already_there_msg(folder)
    else:
        _log("Files successfully downloaded to %s" % folder)


def _make_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                  md5_list=None, doc="", data_size=None, msg=None,
                  unzip=False):
    """ Create a new fetcher

    Parameters
    ----------
    name : str
        The name of the fetcher function.
    folder : str
        The full path to the folder in which the files would be placed locally.
        Typically, this is something like 'pjoin(dipy_home, 'foo')'
    baseurl : str
        The URL from which this fetcher reads files
    remote_fnames : list of strings
        The names of the files in the baseurl location
    local_fnames : list of strings
        The names of the files to be saved on the local filesystem
    md5_list : list of strings, optional
        The md5 checksums of the files. Used to verify the content of the
        files. Default: None, skipping checking md5.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional.
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.
    returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs
    """

    def fetcher():
        files = {}
        for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (baseurl + f, md5_list[i] if
            md5_list is not None else None)
        fetch_data(files, folder, data_size)

        if msg is not None:
            _log(msg)
        if unzip:
            for f in local_fnames:
                split_ext = op.splitext(f)
                if split_ext[-1] == '.gz' or split_ext[-1] == '.bz2':
                    if op.splitext(split_ext[0])[-1] == '.tar':
                        ar = tarfile.open(pjoin(folder, f))
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError('File extension is not recognized')
                elif split_ext[-1] == '.zip':
                    z = zipfile.ZipFile(pjoin(folder, f), 'r')
                    files[f] += (tuple(z.namelist()),)
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError('File extension is not recognized')

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


fetch_test = _make_fetcher(
    "fetch_test",
    pjoin(home, 'test'),
    ANURA_PUBLIC_URL,
    ['lv_0_20221201072652.mp4'],
    ['lv_0_20221201072652.mp4'],
    ['84704a5b5ce57a36b4c391b7edacb4c7'],
    doc="Download the TEST dataset",
    unzip=True)

fetch_anuraset_v2 = _make_fetcher(
    "fetch_anuraset_v2",
    pjoin(home, 'anuraset_v2'),
    ANURA_PUBLIC_URL,
    ['datasetv2-multiclass_1.zip'],
    ['datasetv2-multiclass_1.zip'],
    ['e1fc835fac6ee7973ee307ae64a584e6'],
    doc="Download the ANURA dataset version 2",
    unzip=True)

fetch_anuraset_v3 = _make_fetcher(
    "fetch_anuraset_v3",
    pjoin(home, 'anuraset_v3'),
    ANURA_PUBLIC_URL,
    ['anurasetv3.zip'],
    ['anurasetv3.zip'],
    ['64c86512f338b1a0cf2ff2d4c603b30a'],
    doc="Download the ANURA dataset version 3",
    unzip=True)

# Example to download the anuraset v2 by using the default fetcher
#'''
fetch_anuraset_v3()
#'''

# Example to download the anuraset v2 using a customized fetcher
'''
custom_fetcher = _make_fetcher(
    "fetch_anuraset_v2",
    "/home/jrudascas/Escritorio/VonHumboldt/chorus_experiments/datos_test/anuraset",
    ANURA_PUBLIC_URL,
    ['datasetv2-multiclass_1.zip'],
    ['datasetv2-multiclass_1.zip'],
    ['e1fc835fac6ee7973ee307ae64a584e6'],
    doc="Download the ANURA dataset by using a custome fetcher")

custom_fetcher()
'''