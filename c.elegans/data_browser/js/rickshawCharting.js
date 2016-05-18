/*
    Defines the charting functionality for the Nematodes data browser page.

    Author: Evan Story
    Date created: 20150118
*/

                        // [
                        //         {
                        //                 name: "Northeast",
                        //                 data: [ { x: -1893456000, y: 25868573 }, { x: -1577923200, y: 29662053 }, { x: -1262304000, y: 34427091 }, { x: -946771200, y: 35976777 }, { x: -631152000, y: 39477986 }, { x: -315619200, y: 44677819 }, { x: 0, y: 49040703 }, { x: 315532800, y: 49135283 }, { x: 631152000, y: 50809229 }, { x: 946684800, y: 53594378 }, { x: 1262304000, y: 55317240 } ],
                        //                 color: palette.color()
                        //         },
                        //         {
                        //                 name: "Midwest",
                        //                 data: [ { x: -1893456000, y: 29888542 }, { x: -1577923200, y: 34019792 }, { x: -1262304000, y: 38594100 }, { x: -946771200, y: 40143332 }, { x: -631152000, y: 44460762 }, { x: -315619200, y: 51619139 }, { x: 0, y: 56571663 }, { x: 315532800, y: 58865670 }, { x: 631152000, y: 59668632 }, { x: 946684800, y: 64392776 }, { x: 1262304000, y: 66927001 } ],
                        //                 color: palette.color()
                        //         },
                        //         {
                        //                 name: "South",
                        //                 data: [ { x: -1893456000, y: 29389330 }, { x: -1577923200, y: 33125803 }, { x: -1262304000, y: 37857633 }, { x: -946771200, y: 41665901 }, { x: -631152000, y: 47197088 }, { x: -315619200, y: 54973113 }, { x: 0, y: 62795367 }, { x: 315532800, y: 75372362 }, { x: 631152000, y: 85445930 }, { x: 946684800, y: 100236820 }, { x: 1262304000, y: 114555744 } ],
                        //                 color: palette.color()
                        //         },
                        //         {
                        //                 name: "West",
                        //                 data: [ { x: -1893456000, y: 7082086 }, { x: -1577923200, y: 9213920 }, { x: -1262304000, y: 12323836 }, { x: -946771200, y: 14379119 }, { x: -631152000, y: 20189962 }, { x: -315619200, y: 28053104 }, { x: 0, y: 34804193 }, { x: 315532800, y: 43172490 }, { x: 631152000, y: 52786082 }, { x: 946684800, y: 63197932 }, { x: 1262304000, y: 71945553 } ],
                        //                 color: palette.color()
                        //         }
                        // ]

var rickshawCharting = {
        chartWidth: 700,

        // src: https://github.com/shutterstock/rickshaw/issues/125
        clearGraph: function() {
            // $('#legend').empty();
            $("#contentMain").empty();
            // $('#chart_container').html(
            // '<div id="chart"></div><div id="timeline"></div><div id="slider"></div>'
            // );

            // document.querySelector("#contentMain")
            //     .appendChild(document.querySelector('template').content);

            $("#contentMain").append('<div id="chart_container"><div id="y_axis"></div><div id="chart"></div><div id="legend"></div><form id="offset_form" class="toggler"><input type="radio" name="offset" id="lines" value="lines" checked><label class="lines" for="lines">lines</label><br><input type="radio" name="offset" id="stack" value="zero"><label class="stack" for="stack">stack</label></form></div>');
        },

        drawMultiLineChart: function (chartDataSeries) {

                // Src: http://code.shutterstock.com/rickshaw/tutorial/introduction.html
                // var palette = new Rickshaw.Color.Palette();

                var graph = new Rickshaw.Graph( {
                        element: document.querySelector("#chart"),
                        width: this.chartWidth,
                        height: window.innerHeight - $("#container-nav").innerHeight(),
                        renderer: 'scatterplot',
                        min: "auto",
                        series: chartDataSeries
                } );
                
                var x_axis = new Rickshaw.Graph.Axis.X( { 
                    graph: graph,
                    tickFormat: Rickshaw.Fixtures.Number.formatKMBT
                } );
                
                var y_axis = new Rickshaw.Graph.Axis.Y( {
                        graph: graph,
                        orientation: 'left',
                        tickFormat: Rickshaw.Fixtures.Number.formatKMBT,
                        element: document.getElementById('y_axis'),
                        min: "auto"
                } );
                
                var legend = new Rickshaw.Graph.Legend( {
                        element: document.querySelector('#legend'),
                        graph: graph
                } );
                
                var offsetForm = document.getElementById('offset_form');
                
                offsetForm.addEventListener('change', function(e) {
                        var offsetMode = e.target.value;
                
                        if (offsetMode == 'lines') {
                                graph.setRenderer('line');
                                graph.offset = 'zero';
                        } else {
                                graph.setRenderer('stack');
                                graph.offset = offsetMode;
                        }       
                        graph.render();
                
                }, false);

                new Rickshaw.Graph.HoverDetail({ graph: graph });
                
                graph.render();
        },


        // data.csv:              minute, timeSinceSomeUnknownStartingPoint, elapsedTimeSincePrevFrame, centroidCoordX, centroidCoordY
        // movementFeatures.csv:  minute, timeSinceSomeUnknownStartingPoint, elapsedTimeSincePrevFrame, centroidCoordX, centroidCoordY, speedSinceLastFrame, accelerationSinceLastFrame, angle, angularVelocitySinceLastFrame
        getDataSeries: function(rowsWithoutNan, dataUrl) {
            var palette = new Rickshaw.Color.Palette();
            // return [
            //   // {
            //   //   name: "timeSinceSomeUnknownStartingPoint",
            //   //   data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[1] }; }),
            //   //   color: palette.color()
            //   // },
            //   // {
            //   //   name: "elapsedTimeSincePrevFrame",
            //   //   data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[2] }; }),
            //   //   color: palette.color()
            //   // },

            //   {
            //     name: "centroidCoordX",
            //     // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[3] }; }),
            //     data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[3] }; }),
            //     color: palette.color()
            //   },
            //   {
            //     name: "centroidCoordY",
            //     // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[4] }; }),
            //     data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[4] }; }),
            //     color: palette.color()
            //   }
            //   ,

            //   {
            //     name: "speedSinceLastFrame",
            //     // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[4] }; }),
            //     data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[5] }; }),
            //     color: palette.color()
            //   },
            //   {
            //     name: "accelerationSinceLastFrame",
            //     // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[4] }; }),
            //     data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[6] }; }),
            //     color: palette.color()
            //   }
            //   // ,
            //   // {
            //   //   name: "angularVelocitySinceLastFrame",
            //   //   // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[4] }; }),
            //   //   data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[8] }; }),
            //   //   color: palette.color()
            //   // }

            // ];

            var cols = getColNames(getDataFileMetadataObjByUrl(dataUrl));
            var i = 0;
            var allSeries = _.map(cols, function(c) {
              var series = {
                name: c,
                // data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[3] }; }),
                data: _.map(rowsWithoutNan, function(r) { return { x: r[0], y: r[i] }; }),
                color: palette.color()
              };
              i++;
              return series;
            });
            return allSeries;
        }
};