import datajoint as dj

dj.config['database.host'] = 'tutorial-db.datajoint.io'
dj.config['database.user'] = 'nhabib'
dj.config['database.password'] = None #removed

dj.config.save_global()
