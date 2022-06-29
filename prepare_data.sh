case $1 in
    cifar100)
    cp -r /dataset/Cifar100 ~
    cp -r /dataset/Cifar10 ~
    ;;

    cityscape)
    cp -r /dataset/Cityscapes ~
    bash ~/Cityscapes/decompress.sh
    ;;

    cyclegan)
    cp -r /dataset/Cityscapes_cyclegan ~
    tar xf ~/Cityscapes_cyclegan/cyclegan_cityscape.tar -C  ~/
    cp /dataset/Cityscapes_cyclegan/fid_stat.A.npz /home/jovyan/cyclegan

    cp -r /dataset/Cityscapes_pix2pix ~
    tar xf ~/Cityscapes_pix2pix/cityscapes.tar.gz -C  ~/Cityscapes_pix2pix
    ;;

    *)
    echo "unknown dataset"
    exit 1
    ;;
esac






