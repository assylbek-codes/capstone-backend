#!/bin/bash

# Database credentials
DB_NAME="warehouse_platform"
DB_USER="postgres"
CONTAINER_NAME="capstone-backend_postgres_1"

# Backup directory
BACKUP_DIR="/root/db_backups"
mkdir -p $BACKUP_DIR

# Timestamp for backup filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql"

# Create backup
echo "Creating backup of $DB_NAME database..."
docker exec $CONTAINER_NAME pg_dump -U $DB_USER $DB_NAME > $BACKUP_FILE

# Compress the backup
gzip $BACKUP_FILE

# Keep only the most recent 7 backups
find $BACKUP_DIR -name "${DB_NAME}_*.sql.gz" -type f -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}.gz" 