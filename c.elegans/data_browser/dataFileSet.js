/*
  Contains functions for working with data files.
*/

  // Parse the movementFeatures.csv file into a data structure usable by Rickshaw.Graph.
  // 
  // EX: 2 rows from N2_nf4/data/movementFeatures.csv. Each row comes to this function as part of a single-line string, delimited by line endings.:
  //    8,3.376,0.0259999999999998,NaN,NaN
  //    9,3.401,0.02499999999999991,107,187
  function parseDataCsv(dataSet) { var fn = "parseDataCsv";
    var rowStrs = (dataSet.data.split('\n'));
    var rowsWithoutNan = 
      _.map(
        // filter out rows containing non-numeric values
        _.filter(rowStrs, function(s) { return s.length > 0 && s.indexOf('NaN') < 0; }),
        // convert string-formatted numerics to ints
        function(r) { 
          var rStrs = r.split(',');
          var rowNums = _.map(rStrs, function(r) {
            var pr = parseFloat(r);
            return pr;
          });
          return rowNums;
        }
      );

    console.log('parseDataCsv: rowsWithoutNan:');
    console.log(rowsWithoutNan);
    return rowsWithoutNan;
  }

  function filterRows(dataSet, rowsWithoutNan, timeSeriesFilterDimType, start, end) { var fn = "filterRows";
        var timeArr = _.map(rowsWithoutNan, function(r) { return r[dataSet.metadata.timeSeriesFilterColIdx[timeSeriesFilterDimType]]; });
        rowsWithoutNan = _.filter(rowsWithoutNan, function(r) {
          return r[dataSet.metadata.timeSeriesFilterColIdx[timeSeriesFilterDimType]] >= start 
              && r[dataSet.metadata.timeSeriesFilterColIdx[timeSeriesFilterDimType]] <= end;
        });
    return rowsWithoutNan;
  }



  function getDataFileMetadataObjByUrl(url) { var fn = "getDataFileMetadataObjByUrl";
    return _.find(dataFilesMetadataObjs, function(o) { return o.href == url; });
  }


  function getColNames(dataFileMetadataObj) { var fn = "getColNames";
    var colNames = dataFileMetadataObj.dataSeriesNames.split(',');
    var arr = _.map(colNames, function(n) { return n.trim(); });
    return arr;
  }

