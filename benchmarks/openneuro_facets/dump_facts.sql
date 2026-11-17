-- One JSON line per dataset with the facts compare.py checks answers against.
-- Run read-only against the BIDS-Eye database, e.g. on the server:
--   docker exec -i -e PGOPTIONS="-c default_transaction_read_only=on" bids_postgres \
--     sh -c 'psql -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f -' < dump_facts.sql > db_facts.jsonl
with o as (
  select dataset_id,
    array_agg(distinct datatype) filter (where datatype is not null) dts,
    array_agg(distinct task) filter (where task is not null) tasks,
    array_agg(distinct substring(filename from 'task-([A-Za-z0-9]+)')) filter (where filename ~ 'task-') raw_tasks,
    count(distinct subject) n_sub,
    array_agg(distinct other_entities->>'TracerName') filter (where datatype='positron_emission_tomography' and other_entities->>'TracerName' is not null) tracer,
    array_agg(distinct other_entities->>'TracerRadionuclide') filter (where datatype='positron_emission_tomography' and other_entities->>'TracerRadionuclide' is not null) nuclide,
    array_agg(distinct other_entities->>'Manufacturer') filter (where datatype='positron_emission_tomography' and other_entities->>'Manufacturer' is not null) pet_manu
  from bids_objects group by dataset_id),
p as (
  select dataset_id, count(*) n_part,
    array_agg(distinct diagnosis) filter (where diagnosis is not null) dx,
    array_agg(distinct sex) filter (where sex is not null) sexes,
    array_agg(distinct round(age::numeric,1)) filter (where age is not null) ages
  from bids_participants group by dataset_id)
select json_build_object('acc', d.accession_id, 'species', d.species, 'dx_status', d.dx_status,
  'long', d.study_longitudinal, 'domain', d.study_domain, 'age_min', d.age_min, 'age_max', d.age_max,
  'publish', d.publish_date, 'authors', d.authors, 'senior', d.senior_author, 'name', d.name,
  'dts', o.dts, 'tasks', o.tasks, 'raw_tasks', o.raw_tasks, 'n_sub', o.n_sub, 'tracer', o.tracer, 'nuclide', o.nuclide, 'pet_manu', o.pet_manu,
  'n_part', p.n_part, 'dx', p.dx, 'sexes', p.sexes, 'ages', p.ages)
from bids_datasets d left join o on o.dataset_id=d.id left join p on p.dataset_id=d.id;
