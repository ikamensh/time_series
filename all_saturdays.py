from datetime import date, timedelta

def all_saturdays(year):
   d = date(year, 1, 1)                    # January 1st
   d += timedelta(days = 5 - d.weekday())  # First Sunday

   if d.year != year:
       d += timedelta(days=7)

   while d.year == year:
      yield d
      d += timedelta(days = 7)

for d in all_saturdays(2017):
   print(d)



