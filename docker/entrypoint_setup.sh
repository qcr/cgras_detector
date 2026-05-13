#! /bin/bash

# echo ${ROS_DISTRO}
# echo ${ROS_MASTER}
echo "Setup ROS_MASTER_URI"

# Mount the external DATA drive and expose the 2024 season sorted datasets.
# /dev/disk/by-uuid symlink is stable across reboots; the drive is NTFS (ntfs3 kernel driver).
_DATA_UUID="4850FB7450FB675A"
_DATA_MOUNT="/media/dtsai/DATA"
_SORTED_ROOT="${_DATA_MOUNT}/cgras_datasets/cgras_2024_aims_camera_trolley_fixed_filenames_season_tile_sorted"
_SEASONS="2024_nov 2024_oct"

echo "> Mounting DATA drive (UUID ${_DATA_UUID})"
sudo mkdir -p "${_DATA_MOUNT}"
if ! mountpoint -q "${_DATA_MOUNT}"; then
    sudo mount -t ntfs3 -o ro,uid="$(id -u)",gid="$(id -g)" \
        "/dev/disk/by-uuid/${_DATA_UUID}" "${_DATA_MOUNT}" \
        && echo "  DATA drive mounted at ${_DATA_MOUNT}" \
        || echo "  WARNING: DATA drive mount failed — season data will be unavailable"
fi

for _season in ${_SEASONS}; do
    _target="/home/qcr/cgras_data/Source/${_season}"
    sudo mkdir -p "${_target}"
    if ! mountpoint -q "${_target}"; then
        sudo mount --bind "${_SORTED_ROOT}/${_season}" "${_target}" \
            && echo "  Mounted ${_season} -> ${_target}" \
            || echo "  WARNING: bind mount failed for ${_season}"
    fi
done
unset _DATA_UUID _DATA_MOUNT _SORTED_ROOT _SEASONS _season _target

alias python3=python3.11
# echo ${CATKIN_WS}

echo "> Setting up ROS"
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# if ! [[ -z "$CATKIN_WS" ]]; then
#     echo "> Setting up catkin workspace"
#     source "${CATKIN_WS}/devel/setup.bash"
# fi

if [[ $ROS_MASTER_URI ]]; then
    echo "ROS_MASTER_URI = ${ROS_MASTER_URI}"
    export ROS_MASTER_URI=${ROS_MASTER_URI}
fi
if [[ $ROS_IP ]]; then
    echo "ROS_IP = ${ROS_IP}"
    export ROS_IP=${ROS_IP}
fi

if [ "$ROS_MASTER" = true ]; then
    echo "> Setting up ROScore"
    /bin/bash -c "roscore || exit 0"
else
    if [[ -z "$ROS_MASTER_URI" ]]; then
        echo -e "\033[0;32mROS_MASTER_URL is not set\033[0m"
    fi
fi

# Run CMD from Dockerfile or the overriding command from docker compose yaml file
echo "> Running $@"
exec "$@"
