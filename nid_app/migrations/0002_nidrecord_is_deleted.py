from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nid_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='nidrecord',
            name='is_deleted',
            field=models.BooleanField(
                default=False,
                db_index=True,
                help_text='Soft-delete flag. Deleted records are hidden from API but kept in DB.',
            ),
        ),
    ]
