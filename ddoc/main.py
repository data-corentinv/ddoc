from ddoc import generate, generate_excel
from optparse import OptionParser

if __name__ == "__main__":
    
    parser = OptionParser()
    parser.add_option("-o", "--out-directory", dest="out_directory",
                  help="output path", type="string")
    parser.add_option("-m", "--metadata-location", dest="metadata_location",
                  help="location of metadata.json file if it exists", type="string")
    parser.add_option("-d", "--data-location", dest="data_location",
                  help="data location", type="string")
    parser.add_option("-a", "--addons", dest="addons",
                  help="addons parameter", type="string")
    parser.add_option("-t", "--type-report", dest="report",
                  help="type report (excel or word)", type="string")

    (options, args) = parser.parse_args()

    # default values
    if not options.out_directory: # if is empty
        options.out_directory = '.'
    if not options.metadata_location:
        options.metadata_location = None
    if not options.addons: 
        options.addons = 'none'

    if options.report == "word":
        generate(df=options.data_location, out_directory=options.out_directory, metadata_location=options.metadata_location, addons=options.addons)
    elif options.report == 'excel': 
        generate_excel(df=options.data_location, out_directory=options.out_directory)
    else:
        raise Exception('Incorrect type report. 2 cases are possible : word or excel')