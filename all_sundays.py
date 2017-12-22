from datetime import date, timedelta

def all_sundays(year, month=1):
   d = date(year, month, 1)                    # January 1st
   d += timedelta(days = 6 - d.weekday())  # First Sunday


   if d.month != month:
       d += timedelta(days=7)

   while d.year == year:
      yield d
      d += timedelta(days = 7)

# for d in all_sundays(2017):
#    print(d)



