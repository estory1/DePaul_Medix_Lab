/*
    Defines the charting functionality for the Nematodes data browser page.

    Author: Evan Story
    Date created: 20150208
*/


/**
  Defines the format of, and returns the value of, a series name for a dataset.
*/
var c3Charting = {

  genSeriesName: function(datasetName, colName) { var fn = "genSeriesName";
    return datasetName + ": " + colName;
  },


  /**
    Gets all the data series in a dataset, structured for C3 charts.
    This means that the structure is basically:

      [ 'series/column label', seriesVal1, seriesVal2, ..., seriesValN ]
  */
  getAllDataSeries: function(rowsWithoutNan, dataUrl) { var fn = "getAllDataSeries";
    // eslog(fn, dataUrl);
    var self = this;

    var metadata = getDataFileMetadataObjByUrl(dataUrl);
    // eslog(fn, 'metadata = ' + JSON.stringify(metadata));
    var cols = getColNames(metadata);
    var i = 0;
    var allSeries = _.map(cols, function(c) {
      var series = _.map(rowsWithoutNan, function(r) { return r[i]; });
      // column name is the first element in the array.
      series.unshift( self.genSeriesName(metadata.name, c) );
      i++;
      return series;
    });
    return allSeries;
  },


  /**
    Draws a multiline C3 chart, given multiple input datasets.

      chartHtmlId = The HTML element ID indicating the render location of the chart.
      dataSetsToDisplay = An array of datasets to be displayed.
      allHiddenSeries = An array of data series to hide by default.
  */
  drawMultiFileMultiLineChart: function(chartHtmlId, dataSetsToDisplay, allHiddenSeries) { var fn = "drawMultiFileMultiLineChart";
    var self = this;
    var multiFileColumns = 
      _.reduce(
        _.map(dataSetsToDisplay, function(ds) {
          return self.getAllDataSeries(ds.filteredRowsWithoutNan, ds.dataSet.metadata.href);
        }),
        function(accumCols, colArr) {
          return accumCols ? accumCols.concat(colArr) : colArr;
        }
      );

    var chart = c3.generate({
        bindto: '#' + chartHtmlId,
        data: {
          columns: multiFileColumns
        },
        grid: { 
          x: { show: true },
          y: { show: true }
        },
        axis: {
          x: {
            label: {
              text: "Time",
              position: "outer-middle"
            }
          },
          y: {
            label: {
              text: "Value",
              position: "outer-middle"
            }
          }
        }
    });

    // eslog(fn, 'chart');
    // console.log(chart);

    // DOESN'T WORK PROPERLY: set the X axis range: http://c3js.org/samples/api_axis_range.html
    var allTimeArrVals = 
      _.reduce(
        _.map(
          dataSetsToDisplay, 
          function(ds) { 
            return ds.filteredTimeArr;
          }
        ),
        function(accumTime, timeArr) {
          return !accumTime ? timeArr : accumTime.concat(timeArr);
        }
      );
    var mn = Math.floor(_.min(allTimeArrVals));
    var mx = Math.ceil(_.max(allTimeArrVals));
    // eslog(fn, 'mn = ' + mn + '; mx = ' + mx);
    // chart.axis.range({min: {x: mn}, max: {x: mx}});

    // hide commonly ignored series'
    // var metaDataObj = getDataFileMetadataObjByUrl(dataUrl);
    // var hiddenSeries = _.map(metaDataObj.dataSeriesNamesDefaultHidden.split(','), function(n) { return n.trim(); });

    var hiddenSeries =
      _.reduce(
        _.map(dataSetsToDisplay, function(ds) {
          var hiddenSeriesForDs = _.map(ds.dataSet.metadata.dataSeriesNamesDefaultHidden.split(','), function(n) { return n.trim(); });
          var hiddenSeriesForDsDisplayNames = _.map(hiddenSeriesForDs, function(colName) { return self.genSeriesName(ds.dataSet.metadata.name, colName); });
          // ds.hiddenSeriesForDsDisplayNames = hiddenSeriesForDsDisplayNames;

          // If this dataset has been seen before, then probably not all series will be hidden, implying the user has hidden a series, and so their preferences have been revealed and we don't want to override them.
          // In that case, don't change the set of series being hidden - just accept them as-is by returning an empty array and moving on to the next dataset.
          // eslog(fn, '1: hiddenSeriesForDsDisplayNames');
          // console.log(hiddenSeriesForDsDisplayNames);
          var inter = (_.intersection(allHiddenSeries, hiddenSeriesForDsDisplayNames));
          // eslog(fn, '2: inter');
          // console.log(inter);
          var hiddenSeriesForDisplayNamesToAppend = inter.length > 0 ? [] : hiddenSeriesForDsDisplayNames;

          // eslog(fn, '3: hiddenSeriesForDisplayNamesToAppend');
          // console.log(hiddenSeriesForDisplayNamesToAppend);
          return hiddenSeriesForDisplayNamesToAppend;
        }),
        function(accumHidden, hiddenArr) {
          return !accumHidden ? hiddenArr : accumHidden.concat(hiddenArr);
        }
      );

    // eslog(fn, '4: allHiddenSeries');
    // console.log(allHiddenSeries);
    // eslog(fn, '5 (ret)');
    // console.log(hiddenSeries);
    allHiddenSeries = allHiddenSeries.concat(hiddenSeries);

    // Hide the selected and default series'.
    chart.hide(allHiddenSeries);
  },



  /**
    Draws a multiline chart. Deprecated.
  */
  drawMultiLineChart: function(chartHtmlId, rowsWithoutNan, filteredTimeArr, dataUrl) { var fn = "drawMultiLineChart";
    var chart = c3.generate({
        bindto: '#' + chartHtmlId,
        data: {
          columns: this.getAllDataSeries(rowsWithoutNan, dataUrl)
        },
        grid: { 
          x: { show: true },
          y: { show: true }
        },
        axis: {
          x: {
            label: {
              text: "Time",
              position: "outer-middle"
            }
          },
          y: {
            label: {
              text: "Value",
              position: "outer-middle"
            }
          }
        }
    });

    // DOESN'T WORK PROPERLY: set the X axis range: http://c3js.org/samples/api_axis_range.html
    var mn = Math.floor(_.min(filteredTimeArr));
    var mx = Math.ceil(_.max(filteredTimeArr));
    console.log('drawMultiLineChart: mn = ' + mn + '; mx = ' + mx);
    // chart.axis.range({min: {x: mn}, max: {x: mx}});

    // hide commonly ignored series'
    var metaDataObj = getDataFileMetadataObjByUrl(dataUrl);
    var hiddenSeries = _.map(metaDataObj.dataSeriesNamesDefaultHidden.split(','), function(n) { return n.trim(); });
    chart.hide(hiddenSeries);
  }

};