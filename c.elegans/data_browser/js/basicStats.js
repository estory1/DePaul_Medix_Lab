
      // http://www.endmemo.com/program/js/jstatistics.php
      //Check whether is a number or not
      function isNum(args)
      {
          args = args.toString();

          if (args.length == 0) return false;

          for (var i = 0;  i<args.length;  i++)
          {
              if ((args.substring(i,i+1) < "0" || args.substring(i, i+1) > "9") && args.substring(i, i+1) != "."&& args.substring(i, i+1) != "-")
              {
                  return false;
              }
          }

          return true;
      }


      function variance(arr)
      {
          var len = 0;
          var sum=0;
          for(var i=0;i<arr.length;i++)
          {
                if (arr[i] == ""){}
                else if (!isNum(arr[i]))
                {
                    // alert(arr[i] + " is not number, Variance Calculation failed!");
                    console.error(arr[i] + " is not number, Variance Calculation failed!");
                    return 0;
                }
                else
                {
                   len = len + 1;
                   sum = sum + parseFloat(arr[i]); 
                }
          }

          var v = 0;
          if (len > 1)
          {
              var mean = sum / len;
              for(var i=0;i<arr.length;i++)
              {
                    if (arr[i] == ""){}
                    else
                    {
                        v = v + (arr[i] - mean) * (arr[i] - mean);              
                    }        
              }
              
              return v / len;
          }
          else
          {
               return 0;
          }    
      }


      function median(arr)
      {
          arr.sort(function(a,b){return a-b});
          
          var median = 0;
          
          if (arr.length % 2 == 1)
          {
              median = arr[(arr.length+1)/2 - 1];
          }
          else
          {
              median = (1 * arr[arr.length/2 - 1] + 1 * arr[arr.length/2] )/2;
          }
          
          return median
      }


      function sdAndSE(arr) {
        //Standard deviation
        var sd = Math.sqrt(variance(arr));

        //Standard error
        var se = Math.sqrt(variance(arr)/(arr.length-1));

        return [sd, se];
      }


      function mean(arr) {
        var s = 0;
        for(var i=0; i < arr.length; i++) {
          s += arr[i];
        }
        return (s / arr.length);
      }


      function signalToNoiseRatio(arr) {
        var m = mean(arr);
        var sd = (sdAndSE(arr))[0];
        return (m/sd);
      }
