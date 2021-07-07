

drop table if exists #IdNumbers;
select 
		*
		,	row_number() over (order by IdNumber) as globalID
into 
		#IdNumbers
from
		(
		select 1 as IdNumber
		union
		select 2 as IdNumber
		union 
		select 3 as IdNumber
		union 
		select 4 as IdNumber
		union n
		select 5 as IdNumber
		union 
		select 6 as IdNumber
		union 
		select 7 as IdNumber
		union 
		select 8 as IdNumber
		union 
		select 9 as IdNumber
		union 
		select 10 as IdNumber
		union 
		select 11 as IdNumber
		union 
		select 12 as IdNumber
		union 
		select 13 as IdNumber
		union 
		select 14 as IdNumber
		union 
		select 15 as IdNumber
		union 
		select 16 as IdNumber
	)
	IdNumbers
;

select * from #IdNumbers;

drop table if exists #Emails;
select 
		*
		,	row_number() over (order by Email) + (select max(globalID) from #IdNumbers) as globalID
into 
		#Emails
from
		(
		select 'a' as Email
		union
		select 'b' as Email
		union 
		select 'c' as Email
		union 
		select 'd' as Email
		union 
		select 'e' as Email
		union 
		select 'f' as Email
		union 
		select 'g' as Email
		union 
		select 'h' as Email
		union 
		select 'i' as Email
		union 
		select 'j' as Email
		union 
		select 'k' as Email
		union 
		select 'l' as Email
		union 
		select 'm' as Email
		union 
		select 'n' as Email
		union 
		select 'o' as Email
	)
	Emails
;

select * from #Emails;

drop table if exists #CustomerIds;
select 
		*
		,	row_number() over (order by CustomerId) + (select max(globalID) from #Emails) as globalID
into 
		#CustomerIds
from
		(
		select 'A' as CustomerId
		union
		select 'B' as CustomerId
		union 
		select 'C' as CustomerId
		union 
		select 'D' as CustomerId
		union 
		select 'E' as CustomerId
		union 
		select 'F' as CustomerId
		union 
		select 'G' as CustomerId
		union 
		select 'H' as CustomerId
		union 
		select 'I' as CustomerId
		union 
		select 'J' as CustomerId
		union 
		select 'K' as CustomerId
		union 
		select 'L' as CustomerId
		union 
		select 'M' as CustomerId
		union 
		select 'N' as CustomerId
		union 
		select 'O' as CustomerId
		union
		select 'P' as CustomerId
	)
	CustomerIds
;

select * from #CustomerIds;

drop table if exists #link_IdNumber_Email;
select 
		*
into
		#link_IdNumber_Email
from
		(
			select 2 as IdNumber, 'b' as Email
			union
			select 3 as IdNumber, 'b' as Email
			union 
			select 4 as IdNumber, 'c' as Email
			union
			select 4 as IdNumber, 'd' as Email
			union
			select 5 as IdNumber, 'e' as Email
			union
			select 5 as IdNumber, 'f' as Email
			union
			select 6 as IdNumber, 'e' as Email
			union
			select 6 as IdNumber, 'f' as Email
			union
			select 12 as IdNumber, 'l' as Email
			union
			select 13 as IdNumber, 'm' as Email
			union
			select 14 as IdNumber, 'm' as Email
			union
			select 15 as IdNumber, 'n' as Email
			union
			select 15 as IdNumber, 'o' as Email
			union
			select 16 as IdNumber, 'n' as Email
			union
			select 16 as IdNumber, 'o' as Email
		)
		link_IdNumber_Email
;

drop table if exists #link_IdNumber_CustomerId;
select 
		*
into
		#link_IdNumber_CustomerId
from
		(
			select 7 as IdNumber, 'B' as CustomerId
			union
			select 8 as IdNumber, 'B' as CustomerId
			union
			select 9 as IdNumber, 'C' as CustomerId
			union
			select 9 as IdNumber, 'D' as CustomerId
			union
			select 10 as IdNumber, 'E' as CustomerId
			union
			select 10 as IdNumber, 'F' as CustomerId
			union
			select 11 as IdNumber, 'E' as CustomerId
			union
			select 11 as IdNumber, 'F' as CustomerId
			union
			select 16 as IdNumber, 'P' as CustomerId
		)
		link_IdNumber_CustomerId
;

drop table if exists #link_Email_CustomerId;
select 
		*
into
		#link_Email_CustomerId
from 
		(
			select 'g' as Email, 'G' as CustomerId
			union 
			select 'h' as Email, 'G' as CustomerId
			union 
			select 'i' as Email, 'H' as CustomerId
			union 
			select 'i' as Email, 'I' as CustomerId
			union 
			select 'j' as Email, 'J' as CustomerId
			union 
			select 'j' as Email, 'K' as CustomerId
			union 
			select 'k' as Email, 'J' as CustomerId
			union 
			select 'k' as Email, 'K' as CustomerId
			union 
			select 'l' as Email, 'L' as CustomerId
			union 
			select 'l' as Email, 'M' as CustomerId
			union 
			select 'm' as Email, 'N' as CustomerId
			union 
			select 'n' as Email, 'O' as CustomerId
			union 
			select 'n' as Email, 'P' as CustomerId
			union 
			select 'o' as Email, 'O' as CustomerId
			union 
			select 'o' as Email, 'P' as CustomerId
		)
		link_Email_CustomerId
;


-- update #IdNumbers by left joining to Emails:
drop table if exists #temp_holding_location;
select 
		[IdNumbers].IdNumber
	,	case when [IdNumbers].globalId < [emailGlobalId].globalId or [emailGlobalId].globalId is null then [IdNumbers].globalId else [emailGlobalId].globalId end as 'globalId'
into
		#temp_holding_location
from 
		#IdNumbers
		IdNumbers
left join
		(
			select 
					[link_IdNumber_Email].IdNumber
				,	min([Emails].globalId) as 'globalId'
			from
					#link_IdNumber_Email link_IdNumber_Email
			inner join	
					#Emails Emails
			on
					[link_IdNumber_Email].Email = [Emails].Email
			group by
					[link_IdNumber_Email].IdNumber
		)
		emailGlobalId
on	
		[IdNumbers].IdNumber = [emailGlobalId].IdNumber
;

drop table #IdNumbers;
select
		*
into
		#IdNumbers
from
		#temp_holding_location
;

-- update #Emails by left joining to #IdNumbers:
drop table if exists #temp_holding_location;
select 
		[Emails].Email
	,	case when [Emails].globalId < [IdNumbersGlobalId].globalId or [IdNumbersGlobalId].globalId is null then [Emails].globalId else [IdNumbersGlobalId].globalId end as 'globalId'
into
		#temp_holding_location
from 
		#Emails
		Emails
left join
		(
			select 
					[link_IdNumber_Email].Email
				,	min([IdNumbers].globalId) as 'globalId'
			from
					#link_IdNumber_Email link_IdNumber_Email
			inner join	
					#IdNumbers IdNumbers
			on
					[link_IdNumber_Email].IdNumber = [IdNumbers].IdNumber
			group by
					[link_IdNumber_Email].Email
		)
		IdNumbersGlobalId
on	
		[Emails].Email = [IdNumbersGlobalId].Email
;

drop table #Emails;
select
		*
into
		#Emails
from
		#temp_holding_location
;

select * from #Emails;


-- update #IdNumbers by left joining to #CustomerIds:
drop table if exists #temp_holding_location;
select 
		[IdNumbers].IdNumber
	,	case when [IdNumbers].globalId < [CustomerIdGlobalId].globalId or [CustomerIdGlobalId].globalId is null then [IdNumbers].globalId else [CustomerIdGlobalId].globalId end as 'globalId'
into
		#temp_holding_location
from 
		#IdNumbers
		IdNumbers
left join
		(
			select 
					[link_IdNumber_CustomerId].IdNumber
				,	min([CustomerIds].globalId) as 'globalId'
			from
					#link_IdNumber_CustomerId link_IdNumber_CustomerId
			inner join	
					#CustomerIds CustomerIds
			on
					[link_IdNumber_CustomerId].CustomerId = [CustomerIds].CustomerId
			group by
					[link_IdNumber_CustomerId].IdNumber
		)
		CustomerIdGlobalId
on	
		[IdNumbers].IdNumber = [CustomerIdGlobalId].IdNumber
;

drop table #IdNumbers;
select
		*
into
		#IdNumbers
from
		#temp_holding_location
;

select * from #IdNumbers;

drop table if exists #temp_holding_location;
select 
		[CustomerIds].CustomerId
	,	case when [CustomerIds].globalId < [IdNumbersGlobalId].globalId or [IdNumbersGlobalId].globalId is null then [CustomerIds].globalId else [IdNumbersGlobalId].globalId end as 'globalId'
into
		#temp_holding_location
from 
		#CustomerIds
		CustomerIds
left join
		(
			select 
					[link_IdNumber_CustomerId].CustomerId
				,	min([IdNumbers].globalId) as 'globalId'
			from
					#link_IdNumber_CustomerId link_IdNumber_CustomerId
			inner join	
					#IdNumbers IdNumbers
			on
					[link_IdNumber_CustomerId].IdNumber = [IdNumbers].IdNumber
			group by
					[link_IdNumber_CustomerId].CustomerId
		)
		IdNumbersGlobalId
on	
		[CustomerIds].CustomerId = [IdNumbersGlobalId].CustomerId
;

drop table #CustomerIds;
select
		*
into
		#CustomerIds
from
		#temp_holding_location
;

select * from #CustomerIds;


-- update #IdNumbers by left joining to CustomerIds:
drop table if exists #temp_holding_location;
select 
		[IdNumbers].IdNumber
	,	case when [IdNumbers].globalId < [CustomerIdsGlobalId].globalId or [CustomerIdsGlobalId].globalId is null then [IdNumbers].globalId else [CustomerIdsGlobalId].globalId end as 'globalId'
into
		#temp_holding_location
from 
		#IdNumbers
		IdNumbers
left join
		(
			select 
					[link_IdNumber_CustomerId].IdNumber
				,	min([CustomerIds].globalId) as 'globalId'
			from
					#link_IdNumber_CustomerId link_IdNumber_CustomerId
			inner join	
					#CustomerIds CustomerIds
			on
					[link_IdNumber_CustomerId].CustomerId = [CustomerIds].CustomerId
			group by
					[link_IdNumber_CustomerId].IdNumber
		)
		CustomerIdsGlobalId
on	
		[IdNumbers].IdNumber = [CustomerIdsGlobalId].IdNumber
;

drop table #IdNumbers;
select
		*
into
		#IdNumbers
from
		#temp_holding_location
;

select * from #IdNumbers;

-- update #CustomerIds by left joining to #IdNumbers:
