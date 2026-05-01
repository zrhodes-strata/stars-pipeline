select
    m:tags:prediction_type::string as prediction_type,
    m:tags:cliententityid::string as entity_id,
    t.*
from datalake_sandbox.public_volume_predictions.dagster_run_details t,
     lateral (select parse_json(t.metadata) as m)
where strata_id = 1921 and collection_label='research-2026-03-06-20260307-055820' order by entity_id asc;


select 
    e.strata_id,
    m:tags:prediction_type::string as prediction_type,
    r.collection_label,
    e.run_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup,
    service_line_id,
    service_line
from datalake_sandbox.public_volume_predictions.volume_prediction_exclusions e left join datalake_sandbox.public_volume_predictions.dagster_run_details r on e.run_id=r.run_id,
    lateral (select parse_json(r.metadata) as m)
where e.strata_id=1921 and e.run_date='2026-03-07'
order by
    cliententityid asc,
    patient_type_rollup_id asc,
    service_line_id asc;


with client_runs as (
select
    run_id,
    strata_id
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where strata_id in (14, 84, 1921)
group by
    run_id,
    strata_id
)
select
    cr.run_id,
    cr.strata_id,
    r.run_id,
    r.strata_id,
    r.job_Name,
    r.collection_id,
    r.collection_label
from client_runs cr left join datalake_sandbox.public_volume_predictions.dagster_run_details r on cr.run_id=r.run_id
order by
    cr.strata_id,
    collection_label;



select * from datalake_sandbox.newton_volume_predictions.runs;

--drop table datalake_sandbox.newton_volume_predictions.runs;
create temporary table datalake_sandbox.newton_volume_predictions.runs as
select
    distinct(run_id)
from datalake_sandbox.public_volume_predictions.dagster_run_details
where collection_label in ('Keck First Run', 'Mary Washington Initial Onboarding', 'Emory Initial Full Model Reuse Run');


select top 1000 * from datalake_sandbox.newton_volume_predictions.midterm_cv;
--drop table datalake_sandbox.newton_volume_predictions.midterm_cv;
create temporary table datalake_sandbox.newton_volume_predictions.midterm_cv as 
select
    'Midterm' as cv_type,
    c.*
from datalake_sandbox.newton_volume_predictions.runs r inner join datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs c on r.run_id=c.run_id;



select top 1000 * from datalake_sandbox.newton_volume_predictions.shortterm_cv;
--drop table datalake_sandbox.newton_volume_predictions.shortterm_cv;
create temporary table datalake_sandbox.newton_volume_predictions.shortterm_cv as 
select
    'Short Term' as cv_type,
    c.*
from datalake_sandbox.newton_volume_predictions.runs r inner join datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs c on r.run_id=c.run_id;



select * from datalake_sandbox.newton_volume_predictions.all_cv;
--drop  table datalake_sandbox.newton_volume_predictions.all_cv;
create temporary table datalake_sandbox.newton_volume_predictions.all_cv as
select * from datalake_sandbox.newton_volume_predictions.midterm_cv
union
select * from datalake_sandbox.newton_volume_predictions.shortterm_cv;

--drop table datalake_sandbox.newton_volume_predictions.cv_service_lines;
create temporary table datalake_sandbox.newton_volume_predictions.cv_service_lines as
select 
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date,
    count(distinct(run_id)) as distinct_runs,
    sum(case when prediction is null then 1 else 0 end) as missing_predictions,
    sum(case when actual is null then 1 else 0 end) as missing_actual,
    count(distinct(model_name)) as candidate_models
from datalake_sandbox.newton_volume_predictions.all_cv
group by
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date; --Check to ensure no duplication. There is one run for every service line in the table.




create temporary table datalake_sandbox.newton_volume_predictions.by_type_segment_model_window_month as
select
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date,
    model_name,
    cutoff_date,
    test_start_date,
    test_end_date,
    calendar_year,
    calendar_month,
    sum(prediction) as predicted,
    sum(actual) as actual
from datalake_sandbox.newton_volume_predictions.all_cv
group by
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date,
    model_name,
    cutoff_date,
    test_start_date,
    test_end_date,
    calendar_year,
    calendar_month;


create temporary table datalake_sandbox.newton_volume_predictions.labeled as
select 
    *,
    100 * abs(actual - predicted) / greatest(100, actual) as esh,
    dense_rank() over (partition by cv_type, strata_id, entity_id, patient_type_rollup_id, service_line_id order by cutoff_date asc) as window_number,
    dense_rank() over (partition by cv_type, strata_id, entity_id, patient_type_rollup_id, service_line_id, cutoff_date order by calendar_year asc, calendar_month asc) as month_number,
from datalake_sandbox.newton_volume_predictions.by_type_segment_model_window_month;

create temporary table datalake_sandbox.newton_volume_predictions.by_type_segment_model as
select 
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    model_name,
    training_data_start_date,
    training_data_end_date,
    min(test_start_date) as validation_start_date,
    max(test_end_date) as validation_end_date,
    avg(esh) as mesh,
    avg(actual) as mean_actual,
    avg(predicted) as mean_predicted
from datalake_sandbox.newton_volume_predictions.labeled
group by
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    model_name,
    training_data_start_date,
    training_data_end_date;


create temporary table datalake_sandbox.newton_volume_predictions.champion_cv_results as
select *
from datalake_sandbox.newton_volume_predictions.by_type_segment_model
qualify row_number() over (partition by cv_type, strata_id, entity_id, patient_type_rollup_id, service_line_id order by mesh asc)=1;


select *
from datalake_sandbox.newton_volume_predictions.champion_cv_results;

with by_client as (
select
    cv_type,
    strata_id,
    count(*) as "Service Line Segments",
    round(100 * avg(case when mesh < 10 then 1 else 0 end)) as below_10,
    round(100 * avg(case when mesh < 5 then 1 else 0 end)) as below_5,
    round(100 * avg(case when mesh < 3 then 1 else 0 end)) as below_3
from datalake_sandbox.newton_volume_predictions.champion_cv_results
where service_line_id != '-1' and service_line_clean not in ('zzz_exclude__zzz_exclude', 'other__not_assigned', 'ungroupable__ip_ungroupable', 'documentation', 'all_other__all_other', 'not_specified')
group by
    cv_type,
    strata_id
order by
    cv_type,
    strata_id),
overall as (
select
    cv_type,
    9999 as strata_id,
    count(*) as "Service Line Segments",
    round(100 * avg(case when mesh < 10 then 1 else 0 end)) as below_10,
    round(100 * avg(case when mesh < 5 then 1 else 0 end)) as below_5,
    round(100 * avg(case when mesh < 3 then 1 else 0 end)) as below_3
from datalake_sandbox.newton_volume_predictions.champion_cv_results
where service_line_id != '-1' and service_line_clean not in ('zzz_exclude__zzz_exclude', 'other__not_assigned', 'ungroupable__ip_ungroupable', 'documentation', 'all_other__all_other', 'not_specified')
group by
    cv_type
order by
    cv_type
),
all_results as (
select * from by_client
union
select * from overall
)
select *
from all_results
order by
    case when strata_id=14 then 1 when strata_id=1921 then 2 when strata_id=84 then 3 else 4 end,
    cv_type;



with by_client as (
select
    cv_type,
    strata_id,
    count(*) as "Patient Type Segments",
    round(100 * avg(case when mesh < 10 then 1 else 0 end)) as below_10,
    round(100 * avg(case when mesh < 5 then 1 else 0 end)) as below_5,
    round(100 * avg(case when mesh < 3 then 1 else 0 end)) as below_3
from datalake_sandbox.newton_volume_predictions.champion_cv_results
where service_line_id = '-1' and service_line_clean not in ('zzz_exclude__zzz_exclude', 'other__not_assigned', 'ungroupable__ip_ungroupable', 'documentation', 'all_other__all_other', 'not_specified')
group by
    cv_type,
    strata_id
order by
    cv_type,
    strata_id),
overall as (
select
    cv_type,
    9999 as strata_id,
    count(*) as "Patient Type Segments",
    round(100 * avg(case when mesh < 10 then 1 else 0 end)) as below_10,
    round(100 * avg(case when mesh < 5 then 1 else 0 end)) as below_5,
    round(100 * avg(case when mesh < 3 then 1 else 0 end)) as below_3
from datalake_sandbox.newton_volume_predictions.champion_cv_results
where service_line_id = '-1' and service_line_clean not in ('zzz_exclude__zzz_exclude', 'other__not_assigned', 'ungroupable__ip_ungroupable', 'documentation', 'all_other__all_other', 'not_specified')
group by
    cv_type
order by
    cv_type
),
all_results as (
select * from by_client
union
select * from overall
)
select *
from all_results
order by
    cv_type,
    case when strata_id=14 then 1 when strata_id=1921 then 2 when strata_id=84 then 3 else 4 end;







select * 
from datalake_sandbox.newton_volume_predictions.champion_cv_results
where strata_id=84 and service_line_id='-1'
order by
    entity_id,
    patient_type_rollup_id,
    cv_type;


select *
from datalake_sandbox.newton_volume_predictions.by_type_segment_model_window_month
where strata_id=84 and calendar_year=2026;


select 
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    count(distinct(window_number)) as num_windows
from datalake_sandbox.newton_volume_predictions.labeled
where cv_type='Short Term'
group by
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
order by
    count(distinct(window_number)) asc;



select 
    strata_id,
    cv_type,
    count(*) as service_lines
from datalake_sandbox.newton_volume_predictions.cv_service_lines
group by
    strata_id,
    cv_type
order by
    strata_id,
    cv_type;

select * from datalake_sandbox.newton_volume_predictions.cv_service_lines
where candidate_models != 7; --Service lines from Emory where model was reused

select 
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    max(training_data_end_date) as training_end_date,
    sum(case when cv_type='Midterm' then 1 else 0 end) as midterm,
    sum(case when cv_type='Short Term' then 1 else 0 end) as shortterm,
    sum(case when cv_type='Short Term' then candidate_models else 0 end) as shortterm_candidate_models,
    sum(case when cv_type='Midterm' then candidate_models else 0 end) as midterm_candidate_models
from datalake_sandbox.newton_volume_predictions.cv_service_lines
group by
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
--having sum(case when cv_type='Midterm' then 1 else 0 end)=0
having strata_id=1921 -- and entity_id in (3, 7, 8)
order by 
    strata_id,
    entity_id,
    patient_type_rollup_id,
    service_line_id;

select
    run_id,
    snapshot_date,
    sum(volume)
from datalake_sandbox.public_volume_predictions.client_data_service_line_aggregations_unified
where strata_id=1921 and cliententityid=7 and patient_type_rollup_id=2 and service_line_id='144'
group by
    run_id,
    snapshot_date
order by
    snapshot_date;


select
    run_id,
    snapshot_date,
    sum(volume)
from datalake_sandbox.public_volume_predictions.client_data_service_line_aggregations_unified
where strata_id=1921 and cliententityid=3 and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    snapshot_date
order by
    snapshot_date;

select 
    run_id,
    executed_at
from datalake_sandbox.res_volume_predictions.mid_term_cross_validation_outputs
where strata_id=1921 and entity_id=3 and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;


select 
    run_id,
    executed_at
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where strata_id=1921 and entity_id=3 and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;


select
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
from datalake_sandbox.public_volume_predictions.client_data_service_line_aggregations_unified
where strata_id=1921 and cliententityid=3 and patient_type_rollup_id=2 and service_line_id='131'
limit 1;


select 
    run_id,
    executed_at
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where strata_id=1921 and entity_id=3-- and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;


select 
    run_id,
    executed_at
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where strata_id=1921 and entity_id=10-- and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;


select 
    run_id,
    executed_at
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where strata_id=1921 and entity_id=3 and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;


select 
    run_id,
    executed_at
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where strata_id=1921 and entity_id=3 and patient_type_rollup_id=2 and service_line_id='131'
group by
    run_id,
    executed_at;

select *
from datalake_sandbox.res_volume_predictions.mid_term_cross_validation_outputs
where strata_id=1921 and entity_id=3 and patient_type_rollup_id=2 and service_line_id='131' and run_id='13821aab-ed57-42e2-9588-ef02c6002d16';

select
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date,
    model_name,
    cutoff_date,
    test_start_date,
    test_end_date,
    calendar_year,
    calendar_month,
    sum(prediction) as predicted,
    sum(actual) as actual
from datalake_sandbox.newton_volume_predictions.all_cv
group by
    cv_type,
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    training_data_start_date,
    training_data_end_date,
    model_name,
    cutoff_date,
    test_start_date,
    test_end_date,
    calendar_year,
    calendar_month;


with runs as (
select
    run_id,
    executed_at,
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line,
    sum(volume) as total_volume
from datalake_sandbox.public_volume_predictions.client_data_service_line_aggregations_unified
where strata_id=14
group by
    run_id,
    executed_at,
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line),
pt_level as (
select
    run_id,
    executed_at,
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    count(*) as service_lines,
    sum(total_volume) as total_volume
from runs
where service_line_id != '-1'
group by
    run_id,
    executed_at,
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean
),
entity_level as (
select
    run_id,
    executed_at,
    strata_id,
    cliententityid,
    count(*) as patient_types,
    sum(service_lines) as service_lines,
    sum(total_volume) as total_volume
from pt_level
group by
    run_id,
    executed_at,
    strata_id,
    cliententityid
)
select 
    run_id,
    executed_at,
    strata_id,
    count(*) as entity_count,
    sum(patient_types) as patient_types,
    sum(service_lines) as service_lines,
    sum(total_volume) as total_volume
from entity_level
group by
    run_id,
    executed_at,
    strata_id
order by
    executed_at desc;

show tables in datalake_sandbox.public_volume_predictions;
select top 1000 *
from datalake_sandbox.public_volume_predictions.volume_prediction_exclusions;

select
    entity_id,
    collection_id,
    run_id,
    executed_at,
    count(*) as lines
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where strata_id=14
group by
    entity_id,
    collection_id,
    run_id,
    executed_at
order by
    executed_at desc,
    entity_id asc;

select *
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where collection_id is null and run_id='7a19f939-947b-489a-861b-49a87b8db1e6';


select *
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where strata_id=14 and entity_id=5 and patient_type_rollup_id=3 and service_line_id='90' and run_id='7a19f939-947b-489a-861b-49a87b8db1e6';


select *
from datalake_sandbox.public_volume_predictions.dagster_run_details
where strata_id=84
order by created_at desc;


select *
from datalake_sandbox.public_volume_predictions.dagster_run_details
where strata_id=14
order by created_at desc;


select *
from datalake_sandbox.public_volume_predictions.dagster_run_details
where strata_id=1921
order by created_at desc;


select *
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where strata_id in (14, 1921, 84) and prediction is null;


with runs as (
select
    distinct(run_id)
from datalake_sandbox.public_volume_predictions.dagster_run_details
where collection_label='Keck First Run'
),
midterm_cv as (
select
    'Midterm' as cv_type,
    c.*
from runs r inner join datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs c on r.run_id=c.run_id
),
shortterm_cv as (
select
    'Short term' as cv_type,
    c.*
from runs r inner join datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs c on r.run_id=c.run_id
),
all_cv as (
select * from midterm_cv
union
select * from shortterm_cv
),
by_service_Line as (
select 
    cv_type,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    count(distinct(model_name)) as candidate_models,
    min(test_start_date) as validation_start_date,
    max(test_end_date) as validation_end_date,
    sum(prediction) as total_predicted,
    sum(actual) as total_actual,
    count(*) as lines
from all_cv
group by 
    cv_type,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean)
select
    cv_type,
    count(*) as service_Lines
from by_service_line
group by
    cv_type;
order by count(*) asc;







with shortterm_cv as 
(select *
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where collection_id='3a1cb821-b763-4744-a4de-a116b536538d'),
midterm_cv as (
select *
from datalake_sandbox.public_volume_predictions.mid_term_cross_validation_outputs
where collection_id='3a1cb821-b763-4744-a4de-a116b536538d'
),
shortterm_by_service_line as (
select
    entity_id,
    patient_Type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    count(distinct(model_name)) as candidate_models,
    sum(actual) as total_actual,
    sum(prediction) as total_predicted,
    count(*) as line
from shortterm_cv
group by
    entity_id,
    patient_Type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
),
midterm_by_service_line as (
select
    entity_id,
    patient_Type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    count(distinct(model_name)) as candidate_models,
    sum(actual) as total_actual,
    sum(prediction) as total_predicted,
    count(*) as line
from midterm_cv
group by
    entity_id,
    patient_Type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
)
select * 
from
midterm_by_service_line
order by candidate_models asc;
where prediction is null;
select 
    'Short Term' as cv_type,
    sum(case when service_line_id='-1' then 1 else 0 end) as patient_type_rollups,
    sum(case when service_line_id!='-1' then 1 else 0 end) as service_lines
from shortterm_by_service_line
union
select 
    'Mid Term' as cv_type,
    sum(case when service_line_id='-1' then 1 else 0 end) as patient_type_rollups,
    sum(case when service_line_id!='-1' then 1 else 0 end) as service_lines
from midterm_by_service_line;


select
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    count(*) as lines
from datalake_sandbox.public_volume_predictions.short_term_cross_validation_outputs
where run_id='7a19f939-947b-489a-861b-49a87b8db1e6'
group by
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean
order by
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean;

