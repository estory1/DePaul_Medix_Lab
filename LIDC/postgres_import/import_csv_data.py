import sys
import os

# Relies on csvsql, which I pip installed:
#	pip install csvsql
# whose installation location is: C:\Users\estory\AppData\Local\Continuum\Anaconda2\Scripts\csvsql.exe

pg = {}
pg["user"] = "postgres"
print "Enter the '"+pg["user"]+"' user's password:"
pg["pw"] = (sys.stdin.readline()).strip()   # huh, pw not in a code file on Github, enforcing "something you know" security, fancy that... :-)
pg["host"] = "localhost"
pg["dbname"] = "LIDC_Complete_20141106"


def importCSV(filePath):
	tblname = os.path.basename(filePath)
	tblname = tblname[0:63]					# postgres table names are limited to 64 chars

	print "* Importing " + str(os.path.getsize(filePath)) + " bytes from file: " + filePath

	# Connection string doc: http://docs.sqlalchemy.org/en/latest/core/engines.html
	os.system('csvsql -v --db "postgresql://'+pg["user"]+':'+pg["pw"]+'@'+pg["host"]+'/'+pg["dbname"]+'" --table "'+tblname+'" --insert "' + filePath + '"')


# Import #1
# importCSV("D:\LIDC\LIDC_Complete_20141106\Extracts\imageSOP_UID-filePath-dicominfo-ALL_PATIENTS-with_ConvolutionalKernel_Category.csv")

# Import #2
# importCSV("D:\LIDC\LIDC_Complete_20141106\Extracts\master_join4-no_stmt.csv")

# Import #3
